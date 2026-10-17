"""
Hypergraph Module: cross-modal-only refinement per utterance, sitting after
ContextGraphStack (intra-modality temporal context — see context_graph.py)
and VUEFusion (consensus init — see vue.py). No cross-utterance edges here
at all (GAT already covers temporal), so every utterance's 4-node,
6-hyperedge subgraph is independent and identical in topology — this lets
the whole module be vectorized over [T, B] without any batch_graphify-style
edge_index/scatter machinery.

Per instruction, there is NO input projection (W_in^m / W_in^e from the
spec): the 4 node types are assumed to already share one common dimension
D_h coming in — i.e. context_graph's `hidden_dim` and vue.VUEFusion's
`d_e` must be set equal when wiring this module in (model.py sets
d_e = hidden_dim by default, so this holds automatically). Node values are
used as-is: h_{u_i^m}^(0) = tilde_h_i^m (ContextGraphStack's hidden0_a/t/v),
h_{u_i^e}^(0) = v_hat_i^e (VUEFusion's output).

Also per instruction: no reliability gate (w_R / log w_R) here — the
hyperedge attention score is content-only. (ContextGraphLayer and
VUEFusion each still have their own reliability mechanism; this module
intentionally doesn't.)

Design choices not pinned down by the spec, made consistent with the rest
of this codebase (context_graph.py / vue.py):
  - W_q, W_k, W_v, W_o are shared across all 6 hyperedges within one layer
    (the spec's formulas carry no per-hyperedge-type subscript) — but each
    stacked layer has its OWN copy (layers are not weight-tied, same as
    ContextGraphStack).
  - Degree d_u is a fixed constant per node type (derived from how many of
    the 6 hyperedges contain it: 3 for each modality node, 6 for VUE, since
    every hyperedge contains VUE) — computed once from the hyperedge
    definitions below, not learned or re-derived at runtime.
"""

import torch
import torch.nn as nn


class HypergraphConvLayer(nn.Module):
    """
    One HGConv round (the spec's "2 chặng" — node->hyperedge, then
    hyperedge->node) over the fixed per-utterance hypergraph:

        Nodes:      u_i^a, u_i^t, u_i^v, u_i^e (VUE)
        Hyperedges (6, all containing VUE — "no edge is standalone"):
          Independent: {a,e}, {t,e}, {v,e}         (per-modality vs consensus)
          Coupled:     {a,t,e}, {a,v,e}, {t,v,e}    (pairwise vs consensus)

        q_e          = W_q h_{u_i^e}                (VUE is ALWAYS the query)
        s_{u,e}      = (q_e^T W_k h_u) / sqrt(D_h)     for each u in e (VUE included)
        gamma_{u,e}  = softmax_{u in e}(s_{u,e})
        m_e          = sum_{u in e} gamma_{u,e} (W_v h_u)

        h_u^{(l+1)}  = ReLU( h_u^{(l)} + (1/d_u) sum_{e ni u} W_o m_e )

    forward() returns (h_new, gamma_dict): h_new is the updated 4-node dict,
    gamma_dict[hyperedge_name] is that hyperedge's [T, B, |e|] attention —
    this is the "gamma^hg" the spec earmarks as the distillation target
    (VUE<->modality dynamic agreement), exposed here for a future KD loss
    to consume; this file does not compute any loss itself.
    """

    HYPEREDGES = {
        'ind_a':   ('a', 'e'),
        'ind_t':   ('t', 'e'),
        'ind_v':   ('v', 'e'),
        'coup_at': ('a', 't', 'e'),
        'coup_av': ('a', 'v', 'e'),
        'coup_tv': ('t', 'v', 'e'),
    }
    NODE_TYPES = ('a', 't', 'v', 'e')

    def __init__(self, hidden_dim):
        super(HypergraphConvLayer, self).__init__()
        self.d_h = hidden_dim

        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # node type -> incident hyperedge names, and its (fixed) degree
        self._incident = {
            u: [name for name, members in self.HYPEREDGES.items() if u in members]
            for u in self.NODE_TYPES
        }
        self.degree = {u: len(names) for u, names in self._incident.items()}  # {'a':3,'t':3,'v':3,'e':6}

    def forward(self, h):
        """
        h: {'a'/'t'/'v'/'e': [T, B, hidden_dim]} — current representation
           of each of the 4 node types.

        returns: (h_new, gamma_dict)
          h_new:      same shape/keys as h, this layer's updated nodes.
          gamma_dict: {hyperedge_name: [T, B, |members|]} attention weights,
                      one entry per hyperedge (member order matches
                      HYPEREDGES[hyperedge_name]).
        """
        q_e = self.w_q(h['e'])  # [T, B, D_h] — VUE's own query, reused by every hyperedge

        m_e = {}
        gamma_dict = {}
        for ename, members in self.HYPEREDGES.items():
            scores, values = [], []
            for u in members:
                k_u = self.w_k(h[u])
                score = (q_e * k_u).sum(dim=-1) / (self.d_h ** 0.5)  # [T, B]
                scores.append(score)
                values.append(self.w_v(h[u]))

            scores = torch.stack(scores, dim=-1)      # [T, B, |members|]
            gamma = torch.softmax(scores, dim=-1)
            gamma_dict[ename] = gamma

            values = torch.stack(values, dim=-2)        # [T, B, |members|, D_h]
            m_e[ename] = (gamma.unsqueeze(-1) * values).sum(dim=-2)  # [T, B, D_h]

        h_new = {}
        for u in self.NODE_TYPES:
            agg = sum(self.w_o(m_e[ename]) for ename in self._incident[u]) / self.degree[u]
            h_new[u] = torch.relu(h[u] + agg)

        return h_new, gamma_dict


class HypergraphStack(nn.Module):
    """
    Stacks `n_layers` HypergraphConvLayer rounds (n_layers is a
    hyperparameter). Layers are NOT weight-tied — each has its own
    W_q/W_k/W_v/W_o. Returns the final 4-node dict (h['e'] is the spec's
    final VUE, v_hat_i^{e,(L)}) plus every layer's gamma_dict, in case a
    distillation loss wants intermediate hops too rather than only the
    last one.
    """

    def __init__(self, hidden_dim, n_layers=2):
        super(HypergraphStack, self).__init__()
        assert n_layers >= 1, f'n_layers must be >= 1, got {n_layers}'
        self.layers = nn.ModuleList([
            HypergraphConvLayer(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, h):
        """
        h: {'a'/'t'/'v'/'e': [T, B, hidden_dim]} — initial nodes
           (tilde_h_i^m for a/t/v, v_hat_i^e for e; no projection applied
           here, see module docstring).

        returns: (h, gammas) — h is the final 4-node dict after all layers,
                 gammas is a list of that layer's gamma_dict, one per layer
                 in order.
        """
        gammas = []
        for layer in self.layers:
            h, gamma_dict = layer(h)
            gammas.append(gamma_dict)
        return h, gammas
