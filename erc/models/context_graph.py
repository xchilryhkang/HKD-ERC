import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax

from erc.models.graph import utterance_to_conversation


class ContextGraphLayer(nn.Module):
    """
    Multi-head attention over a temporal window, with three learned edge
    priors added as LOG-priors to the content score (so a prior of 0 kills
    the edge after softmax, and can never go negative):

        w_R(j)     = 1 if node j is present, else placeholder reliability
                     ('feature': sigmoid(MLP(h_j)), 'scalar': one learned knob)
        w_S(i,j)   = 1 if same speaker, else beta
        w_T^k(i,j) = exp(-(t_i - t_j)^2 / (2 sigma_k^2))   one sigma per head

        alpha^k = softmax_j( LeakyReLU((W_q x_i)^T W_k x_j) + log(w_T^k w_S w_R) )
        x_out   = LayerNorm( x_i + W_o concat_k sum_j alpha^k W_v^k x_j )
    """

    def __init__(self, hidden_dim, n_heads=4, dropout=0.5, reliability_type='feature'):
        super(ContextGraphLayer, self).__init__()
        assert hidden_dim % n_heads == 0, (
            f'hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})'
        )
        assert reliability_type in ('scalar', 'feature'), (
            f"reliability_type must be 'scalar' or 'feature', got {reliability_type!r}"
        )
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.reliability_type = reliability_type

        # ---- content attention (per head, packed as one Linear each) ---- #
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ---- learned prior knobs (this layer's own — not shared across depth) ---- #
        self.theta_beta = nn.Parameter(torch.tensor(0.0))   # speaker consistency, shared across heads
        if reliability_type == 'scalar':
            self.theta_rho = nn.Parameter(torch.tensor(0.0))    # single global reliability knob
        else:
            self.reliability_mlp = nn.Sequential(                 # sigmoid(MLP(h_j)) — placeholder reliability
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        self.eta = nn.Parameter(torch.zeros(n_heads))        # one temporal scale per head

        self.attn_dropout = nn.Dropout(dropout)

        # ---- output projection (after concatenating heads) + residual path ---- #
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features, edge_index, features_speaker_ids, features_node_positions, availability):
        """
        features     [N, hidden_dim]   edge_index [2, E] (row 0 = src j, row 1 = dst i)
        availability [N]  1 = modality present at that node, 0 = placeholder.
                     Describes the raw input, so it is the same at every hop.
        returns      x_out [N, hidden_dim]
        """
        N = features.size(0)
        src, dst = edge_index[0], edge_index[1]

        availability = availability.to(features.dtype)

        beta = torch.sigmoid(self.theta_beta)
        same_speaker = (features_speaker_ids[src] == features_speaker_ids[dst]).to(features.dtype)
        w_S = same_speaker + (1 - same_speaker) * beta  # [E]

        # Reliability is computed once per NODE then gathered per edge, which
        # avoids running the MLP once per edge (edges far outnumber nodes).
        if self.reliability_type == 'scalar':
            reliability = torch.sigmoid(self.theta_rho).expand(N)
        else:
            reliability = torch.sigmoid(self.reliability_mlp(features)).squeeze(-1)
        w_R_node = availability + (1 - availability) * reliability  # 1 if present
        w_R = w_R_node[src]  # [E]

        sigma = F.softplus(self.eta)  # [K], > 0
        dt2 = (features_node_positions[src] - features_node_positions[dst]).to(features.dtype) ** 2  # [E]
        w_T = torch.exp(-dt2.unsqueeze(-1) / (2 * sigma.unsqueeze(0) ** 2))  # [E, K]

        prior = w_T * (w_S * w_R).unsqueeze(-1)  # [E, K]
        log_prior = torch.log(prior.clamp(min=1e-8))

        Q = self.w_q(features).view(N, self.n_heads, self.head_dim)
        K = self.w_k(features).view(N, self.n_heads, self.head_dim)
        V = self.w_v(features).view(N, self.n_heads, self.head_dim)

        content = F.leaky_relu((Q[dst] * K[src]).sum(dim=-1))  # [E, K]
        logit = content + log_prior                             # [E, K]

        alpha = scatter_softmax(logit, dst, num_nodes=N)          # [E, K], normalized per destination
        alpha = self.attn_dropout(alpha)

        weighted_v = alpha.unsqueeze(-1) * V[src]  # [E, K, head_dim]
        agg = torch.zeros(N, self.n_heads, self.head_dim, device=features.device, dtype=features.dtype)
        agg.index_add_(0, dst, weighted_v)
        h_prime = agg.reshape(N, self.n_heads * self.head_dim)  # concat heads -> [N, hidden_dim]

        # Residual ADD, deliberately not the spec's concat [h_i || h_i'] --
        # concat would double the width and force a second hidden size to be
        # tracked through VUE/Hypergraph; this keeps one hidden_dim throughout.
        x_out = self.norm(features + self.w_o(h_prime))
        return x_out


class ContextGraphStack(nn.Module):
    """`n_layers` rounds of ContextGraphLayer, hidden_dim -> hidden_dim, not weight-tied."""

    def __init__(self, hidden_dim, n_layers=2, n_heads=4, dropout=0.5, no_cuda=False,
                 reliability_type='feature'):
        super(ContextGraphStack, self).__init__()
        assert n_layers >= 1, f'graph_layers must be >= 1, got {n_layers}'
        self.no_cuda = no_cuda
        self.layers = nn.ModuleList([
            ContextGraphLayer(hidden_dim, n_heads, dropout, reliability_type)
            for _ in range(n_layers)
        ])

    def forward(self, features, edge_index, features_speaker_ids, features_node_positions,
                availability, seq_lengths, umask):
        """features [N, hidden_dim] (flat, per modality) -> [T, B, hidden_dim]."""
        x = features
        for layer in self.layers:
            x = layer(x, edge_index, features_speaker_ids, features_node_positions, availability)

        out = utterance_to_conversation(x, seq_lengths, umask, self.no_cuda)  # [T, B, hidden_dim]
        return out
