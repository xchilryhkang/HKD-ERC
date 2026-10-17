"""
Context-Graph Module: per-modality Weighted-GAT with structured edge priors.
See context_graph_module.md for the full derivation; this file implements
sections 2-6 of that spec (section 1 — Proj + Placeholder + BiLSTM — is
already implemented in model.py's proj_a/t/v + prompt.py + lstm_a/t/v,
feeding this module's `features` input), stacked into `graph_layers` rounds
of message passing (graph_layers is a hyperparameter; the spec itself only
describes one round, so ContextGraphStack is the generalization to depth L).

Unified dimension: attention (Q/K/V) and the residual update both operate
directly in `hidden_dim`, split evenly across `n_heads` (head_dim =
hidden_dim // n_heads) — no separate "graph_hidden_size" bottleneck. This
means ContextGraphStack's output is exactly `hidden_dim`, matching its
input, so it composes with VUE/Hypergraph (vue.py / hypergraph.py) without
any extra projection — everything downstream shares this one hidden_dim.
This is a deliberate deviation from the spec image's literal residual
CONCAT ([h_i || h_i'], which would double the width): here the attention
output is projected back to hidden_dim and added residually instead, same
spirit (original representation preserved via the residual path) without
introducing a second width to track.

Three independent stacks are built (one per modality: a / t / v, see
model.py), each with its own weights — nothing in this file is
modality-aware, it just runs whatever it's given.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax

from erc.models.graph import utterance_to_conversation


class ContextGraphLayer(nn.Module):
    """
    One round of multi-head attention over a fixed temporal window, with
    three learned edge priors gating the softmax (added as log-priors to
    the content score, so a prior of 0 drives that edge's post-softmax
    weight to 0 without ever letting it go negative):

        #1 reliability   w_R(j) = 1                  if node j is present
                                = sigmoid(MLP(h_j))   if node j is a placeholder
        #2 speaker cons. w_S(i,j) = 1 if same speaker,   beta   if different
        #3 multi-scale temporal decay (one sigma per head):
             w_T^(k)(i,j) = exp(-(t_i - t_j)^2 / (2 sigma_k^2))

        prior^(k)(i,j) = w_T^(k)(i,j) * w_S(i,j) * w_R(j)
        c^(k)(i,j)     = LeakyReLU((W_q x_i)^T (W_k x_j))
        alpha^(k)(i,j) = softmax_j( c^(k)(i,j) + log prior^(k)(i,j) )
        h_i'           = concat_k sum_j alpha^(k)(i,j) (W_v^(k) x_j)      [N, hidden_dim]
        x_out           = LayerNorm( x_i + W_o h_i' )                      [N, hidden_dim]

    Unlike beta (a single learned scalar, shared across heads), reliability
    for a placeholder node is CONTENT-dependent: a small MLP looks at that
    node's own feature h_j (its projected/prompted representation — see
    prompt.py, this is exactly the value a placeholder got assigned) and
    predicts how much to trust it, in (0, 1) via sigmoid. A present node's
    reliability is hard-fixed at 1, never routed through the MLP — "luôn
    tin đủ", not learned. beta is shared across heads; the reliability MLP's
    output is also shared across heads (no head index in the spec either).
    Every learned knob here (beta, the MLP, sigma, W_q/W_k/W_v/W_o) belongs
    to THIS layer only — layers in a ContextGraphStack are not weight-tied,
    so depth 2 has its own beta/reliability-MLP/sigma at each hop, and since
    each hop's `features` input is the previous hop's refined
    representation, reliability can be re-estimated as the placeholder's
    representation becomes better-informed by context through the stack.

    forward() returns x_out [N, hidden_dim]: h_i' (multi-head attention
    output, hidden_dim-wide) projected through W_o and added residually to
    the input, then layer-normed. This is both what a later layer (if any)
    consumes as its own `features`, and — for the last layer in a stack —
    what ContextGraphStack returns.

    For a full-modality input (nothing missing, availability all 1s — the
    teacher's case, or any student batch with mask_rate=0), w_R(j) is 1
    for every edge — the reliability mechanism is still evaluated
    (vectorized over all nodes for simplicity) but its output is multiplied
    by 0 everywhere, so it gets zero gradient and factor #1 contributes 0
    to the log-prior: the layer reduces to a plain multi-scale
    speaker/temporal GAT, matching section 7 of the spec.

    reliability_type picks HOW the placeholder branch of w_R is computed:
      - 'scalar'  a single learned theta_rho, same value for every
                  placeholder node regardless of its feature (the original,
                  simpler design).
      - 'feature' sigmoid(MLP(h_j)) — content-dependent, looks at that
                  node's own (projected/prompted) feature. Default.
    Either way, present nodes stay hard-fixed at 1, never routed through
    either mechanism.
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
        features:                [N, hidden_dim] — this hop's input node
                                  representation (the original h_i for the
                                  first layer in a stack, the previous
                                  layer's x_out for later ones).
        edge_index:               [2, E] — edge_index[0] = source/neighbor j,
                                  edge_index[1] = destination/center i (see
                                  context_graph_module.md section 2; for the
                                  symmetric windows this codebase always uses,
                                  batch_graphify's edge set is closed under
                                  swapping src/dst, so this labeling is exact).
        features_speaker_ids:     [N] — speaker id per node.
        features_node_positions:  [N] — utterance index within its dialogue.
        availability:             [N] — a_j in {0, 1}: 1 = modality present at
                                  that node, 0 = placeholder. This is the
                                  modality's own column of umask_flat (the
                                  5th value batch_graphify returns), the SAME
                                  every layer — it describes the raw input
                                  data, not the evolving representation.

        returns: x_out [N, hidden_dim]
        """
        N = features.size(0)
        src, dst = edge_index[0], edge_index[1]

        availability = availability.to(features.dtype)

        # ============================ #
        #     #2 speaker consistency (shared across heads)
        # ============================ #
        beta = torch.sigmoid(self.theta_beta)
        same_speaker = (features_speaker_ids[src] == features_speaker_ids[dst]).to(features.dtype)
        w_S = same_speaker + (1 - same_speaker) * beta  # [E]

        # ============================ #
        #     #1 reliability of the source node j — hard-fixed at 1 for
        #     present nodes; for placeholders, either a single learned
        #     scalar ('scalar') or sigmoid(MLP(h_j)) ('feature'), shared
        #     across heads either way. Computed once per NODE (not per
        #     edge) then gathered per edge below — for 'scalar' this is
        #     just a broadcast, for 'feature' it avoids recomputing the
        #     MLP once per edge (nodes are typically far fewer than edges).
        # ============================ #
        if self.reliability_type == 'scalar':
            reliability = torch.sigmoid(self.theta_rho).expand(N)  # [N], same value everywhere
        else:
            reliability = torch.sigmoid(self.reliability_mlp(features)).squeeze(-1)  # [N], sigmoid(MLP(h_j))
        w_R_node = availability + (1 - availability) * reliability  # [N]: 1 if present, reliability if placeholder
        w_R = w_R_node[src]  # [E]

        # ============================ #
        #     #3 multi-scale temporal decay (per head)
        # ============================ #
        sigma = F.softplus(self.eta)  # [K], > 0
        dt2 = (features_node_positions[src] - features_node_positions[dst]).to(features.dtype) ** 2  # [E]
        w_T = torch.exp(-dt2.unsqueeze(-1) / (2 * sigma.unsqueeze(0) ** 2))  # [E, K]

        prior = w_T * (w_S * w_R).unsqueeze(-1)  # [E, K]
        log_prior = torch.log(prior.clamp(min=1e-8))

        # ============================ #
        #     content score + prior-gated softmax
        # ============================ #
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

        x_out = self.norm(features + self.w_o(h_prime))  # [N, hidden_dim]
        return x_out


class ContextGraphStack(nn.Module):
    """
    Stacks `n_layers` ContextGraphLayer rounds of message passing (n_layers
    is a hyperparameter, see model.py's `graph_layers` / --graph-layers).
    Every layer is hidden_dim -> hidden_dim (residual updates, see
    ContextGraphLayer), so the stack's output is simply the last layer's
    x_out — no extra concat/projection needed at the stack level:

        x_0 = features                (the true h_i, from BiLSTM)
        x_1 = layer_1(x_0, ...)
        x_2 = layer_2(x_1, ...)
        ...
        x_L = layer_L(x_{L-1}, ...)
        out_i = x_L

    Layers are NOT weight-tied — each has its own beta/rho/sigma/Q/K/V/W_o,
    so depth lets different hops learn different speaker/temporal/
    reliability trade-offs, standard for stacked GAT/GNN blocks. edge_index,
    speaker ids, node positions and availability are constant across hops
    (they describe the fixed graph and the fixed raw-input missingness, not
    the evolving representation), so they're passed unchanged to every layer.
    """

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
        """
        features: [N, hidden_dim] — flat node features for this modality
                  (batch_graphify's output).
        seq_lengths, umask: forwarded to utterance_to_conversation to
                  reshape the flat [N, ...] result back to [T, B, ...].

        returns: [T, B, hidden_dim]
        """
        x = features
        for layer in self.layers:
            x = layer(x, edge_index, features_speaker_ids, features_node_positions, availability)

        out = utterance_to_conversation(x, seq_lengths, umask, self.no_cuda)  # [T, B, hidden_dim]
        return out
