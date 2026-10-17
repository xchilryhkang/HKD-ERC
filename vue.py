"""
VUE (Virtual Utterance Embedding): attention-fusion over the 3 modalities,
per utterance. See vue_module (spec image) for the derivation. Consumes the
context-graph output per modality — h_i^m = [h_i || h_i'] from
context_graph.ContextGraphStack, i.e. model.py's hidden0_a/t/v, shape
[T, B, concat_dim] — and produces ONE new per-utterance embedding v_hat_i,
independent from a/t/v (it doesn't replace or feed back into any of them).
It is the initialization for the hypergraph stage (see hypergraph.py) that
consumes it; both are wired together in model.HypergraphERCModel.forward.

Deviation from the spec image, per explicit instruction: step 1 drops the
LeakyReLU (z_i^m is a plain affine projection, no activation).
"""

import torch
import torch.nn as nn


class VUEFusion(nn.Module):
    """
        Step 1 (consensus projection, shared across modalities, no activation):
            z_i^m = W_c h_i^m + b_c                              in R^{D_e}

        Step 2 (attention score, optionally gated by the same log-prior
        trick as ContextGraphLayer's reliability — see context_graph.py):
            s_i^m = q_c^T tanh(W_s z_i^m) + log w_R(i, m)     if use_reliability_gate
                  = q_c^T tanh(W_s z_i^m)                      if not use_reliability_gate
            w_R(i, m) = 1              if a_i^m == 1 (modality m present at utterance i)
                      = reliability(z_i^m)  if a_i^m == 0 (placeholder)

        use_reliability_gate (default True) turns the log w_R term on/off
        entirely — when off, no reliability parameter is even created (see
        __init__), and attention is driven by content alone, same as never
        gating the graph's own attention. This flag exists only here for
        now; ContextGraphLayer doesn't have an equivalent yet.

        Same reliability_type choice as ContextGraphLayer (only relevant
        when use_reliability_gate=True):
          - 'scalar'  reliability = sigmoid(theta_rho_e), one global learned
                      value shared by every placeholder, regardless of z_i^m.
          - 'feature' reliability = sigmoid(MLP(z_i^m)) — content-dependent,
                      looks at that utterance-modality's own consensus-space
                      projection. Default.

        Step 3 (normalize across the 3 modalities, per utterance):
            gamma_i^m = softmax_{m' in {a,t,v}}(s_i^{m'})

        Step 4 (VUE embedding):
            v_hat_i = sum_m gamma_i^m * z_i^m                     in R^{D_e}

        q_c, W_s, theta_rho_e are shared across modalities (no modality index
        in the spec) — one consensus query/gate, not one per modality.
        Present utterances are hard-fixed at w_R = 1 for that modality, same
        as ContextGraphLayer: when nothing is missing (availability all 1s,
        e.g. the teacher), theta_rho_e receives zero gradient and this module
        reduces to a plain reliability-free attention fusion, matching the
        "teacher self-cancels the reliability factor" property used
        throughout this codebase (see context_graph.py section 7 note).
    """

    def __init__(self, concat_dim, d_e, modal_keys=('a', 't', 'v'), reliability_type='feature',
                 use_reliability_gate=True):
        super(VUEFusion, self).__init__()
        assert reliability_type in ('scalar', 'feature'), (
            f"reliability_type must be 'scalar' or 'feature', got {reliability_type!r}"
        )
        self.modal_keys = list(modal_keys)
        self.d_e = d_e
        self.reliability_type = reliability_type
        self.use_reliability_gate = use_reliability_gate

        self.w_c = nn.Linear(concat_dim, d_e)          # step 1, shared across modalities
        self.w_s = nn.Linear(d_e, d_e, bias=False)      # step 2
        self.q_c = nn.Parameter(torch.randn(d_e) * (d_e ** -0.5))  # consensus query

        if use_reliability_gate:
            if reliability_type == 'scalar':
                self.theta_rho_e = nn.Parameter(torch.tensor(0.0))          # single global reliability knob
            else:
                self.reliability_mlp = nn.Sequential(                        # sigmoid(MLP(z_i^m)) per utterance-modality
                    nn.Linear(d_e, max(d_e // 2, 1)),
                    nn.ReLU(),
                    nn.Linear(max(d_e // 2, 1), 1),
                )

    def forward(self, h_dict, availability_dict):
        """
        h_dict:            {'a'/'t'/'v': [T, B, concat_dim]} — context-graph
                            output per modality (model.py's hidden0_a/t/v).
        availability_dict: {'a'/'t'/'v': [T, B]} — a_i^m in {0, 1}: 1 if that
                            modality is present at utterance i, 0 if it's a
                            placeholder. This is input_features_mask[0]'s
                            per-modality column ([:, :, 0/1/2] for a/t/v),
                            NOT umask_flat (that's the flat graph-node
                            version of the same thing; VUE runs after the
                            context-graph has already reshaped back to
                            [T, B, ...], so it uses the [T, B] mask directly).

        returns:
            v_hat: [T, B, d_e]    — the new virtual-utterance node.
            gamma: [T, B, M]      — attention weight per modality (M = len(modal_keys)),
                                    exposed for inspection/debugging.
        """
        z = {m: self.w_c(h_dict[m]) for m in self.modal_keys}  # step 1, each [T, B, d_e]

        scores = []
        for m in self.modal_keys:
            content = torch.tanh(self.w_s(z[m])) @ self.q_c  # [T, B]

            if not self.use_reliability_gate:
                scores.append(content)                          # [T, B], content only
                continue

            if self.reliability_type == 'scalar':
                reliability = torch.sigmoid(self.theta_rho_e)              # scalar, broadcasts
            else:
                reliability = torch.sigmoid(self.reliability_mlp(z[m])).squeeze(-1)  # [T, B]

            avail = availability_dict[m].to(z[m].dtype)            # [T, B]
            w_R = avail + (1 - avail) * reliability                  # [T, B]: 1 if present, reliability if placeholder
            log_w_R = torch.log(w_R.clamp(min=1e-8))

            scores.append(content + log_w_R)                    # [T, B]

        scores = torch.stack(scores, dim=-1)      # [T, B, M]
        gamma = torch.softmax(scores, dim=-1)      # step 3, [T, B, M]

        z_stack = torch.stack([z[m] for m in self.modal_keys], dim=-2)  # [T, B, M, d_e]
        v_hat = (gamma.unsqueeze(-1) * z_stack).sum(dim=-2)              # step 4, [T, B, d_e]

        return v_hat, gamma
