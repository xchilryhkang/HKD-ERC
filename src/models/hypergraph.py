import torch
import torch.nn as nn


class HypergraphConvLayer(nn.Module):
    """
    One round of node -> hyperedge -> node message passing over the fixed
    per-utterance hypergraph of 4 nodes (a, t, v, e) and 6 hyperedges. Every
    hyperedge contains the consensus node e, which is always the query:

        gamma_{u,e} = softmax_{u in e}( (W_q h_e)^T W_k h_u / sqrt(D_h) )
        m_e         = sum_{u in e} gamma_{u,e} W_v h_u
        h_u'        = ReLU( h_u + (1/deg u) sum_{e ni u} W_o m_e )
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
        h {'a','t','v','e': [T, B, hidden_dim]} -> (h_new, gamma_dict), where
        gamma_dict[edge] is [T, B, |members|] in HYPEREDGES[edge] order.
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
    """`n_layers` rounds of HypergraphConvLayer, not weight-tied."""

    def __init__(self, hidden_dim, n_layers=2):
        super(HypergraphStack, self).__init__()
        assert n_layers >= 1, f'n_layers must be >= 1, got {n_layers}'
        self.layers = nn.ModuleList([
            HypergraphConvLayer(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, h):
        """h (initial 4 nodes) -> (final h, one gamma_dict per layer)."""
        gammas = []
        for layer in self.layers:
            h, gamma_dict = layer(h)
            gammas.append(gamma_dict)
        return h, gammas
