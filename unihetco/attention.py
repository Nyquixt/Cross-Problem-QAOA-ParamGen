import torch.nn as nn
import torch.nn.functional as F
import torch
# from SAWT

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        x=self.activation(x)
        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, atten_mode = "softmax"):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.atten_mode = atten_mode

    def sinkhorn(self, tensors):
        tensors = F.sigmoid(tensors)

        for _ in range(5):
            row_sum = torch.sum(tensors, dim= 3).unsqueeze(3) + (1e-9)
            tensors = tensors / row_sum
            col_sum = torch.sum(tensors, dim= 2).unsqueeze(2) + (1e-9)
            tensors = tensors / col_sum

        return tensors

    def forward(self, q, k, v, cost_mat=None, mask=None):
        #Q K V [batchsize,num_head,len,dim]   the output is the same.
        if cost_mat is not None:
            cost_mat_score = cost_mat[:, None, :, :].expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        if self.atten_mode == "softmax":
            attn = self.dropout(F.softmax(attn, dim=-1))
        else:
            attn = self.sinkhorn(attn)

        if mask is not None:
            attn = attn * mask

        # import pdb; pdb.set_trace()
        if cost_mat is not None:
            attn = attn*cost_mat_score
        
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttentionMixedScore(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, atten_mode = "softmax"):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, atten_mode = atten_mode)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, cost_mat = None, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, cost_mat=cost_mat, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn

class MixEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(MixEncoderLayer, self).__init__()
        self.head_num = n_head

        self.Wq = nn.Linear(d_model, n_head * d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_head * d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_head * d_v, bias=False)
        self.mix_attn = MultiHeadAttentionMixedScore(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, cost_mat=None ,k_v=None):
        enc_input = enc_input
        # import pdb; pdb.set_trace()

        if k_v == None:
            enc_output, _ = self.mix_attn(
                enc_input, enc_input, enc_input, 
                cost_mat
            )
        else:
            enc_output,_ = self.mix_attn(
                enc_input, k_v, k_v,
                cost_mat
            )

        enc_output = self.pos_ffn(enc_output)
        return enc_output

def prepare_attention_data(
    x_var, x_constr, 
    batch_var, batch_constr,
    edge_index
):
    device = x_var.device
    d_model = x_var.size(1) # hidden size

    # per-graph sizes
    B = int(batch_var.max().item() + 1) if batch_var.numel() > 0 else 1
    n_vars = torch.bincount(batch_var, minlength=B) # [B]
    m_constr = torch.bincount(batch_constr, minlength=B) # [B]

    max_n = int(n_vars.max().item()) if B > 0 else 0
    max_m = int(m_constr.max().item()) if B > 0 else 0

    # pack decision & constraint node features into padded batches
    Q_dec = x_var.new_zeros(B, max_n, d_model) # [B, max_N, F]
    K_constr = x_constr.new_zeros(B, max_m, d_model) # [B, max_M, F]

    for b in range(B):
        idx_v = (batch_var == b).nonzero(as_tuple=True)[0]
        idx_c = (batch_constr == b).nonzero(as_tuple=True)[0]

        n_b = idx_v.numel()
        m_b = idx_c.numel()

        if n_b > 0:
            Q_dec[b, :n_b] = x_var[idx_v]
        if m_b > 0:
            K_constr[b, :m_b] = x_constr[idx_c]

    edge_index = edge_index.to(device)  # [2, E]
    row_c = edge_index[0]  # constraint node indices (global in batch space)
    col_v = edge_index[1]  # variable node indices (global in batch space)

    # graph id for each edge (same from constr and var)
    g = batch_constr[row_c]  # [E], each in [0, B-1]

    # offsets so we can turn global indices into local (per-graph) indices
    # constr_indices for graph b run from constr_offsets[b] .. + m_constr[b]-1
    constr_offsets = torch.zeros(B, dtype=torch.long, device=device)
    var_offsets = torch.zeros(B, dtype=torch.long, device=device)
    constr_offsets[1:] = m_constr.cumsum(0)[:-1]
    var_offsets[1:] = n_vars.cumsum(0)[:-1]

    local_c = row_c - constr_offsets[g]  # [E], in [0 .. m_constr[g]-1]
    local_v = col_v - var_offsets[g]     # [E], in [0 .. n_vars[g]-1]

    cost_mat = Q_dec.new_zeros(B, max_n, max_m)  # [B, max_N, max_M]
    cost_mat[g, local_v, local_c] = 1.0

    return Q_dec, K_constr, cost_mat
    