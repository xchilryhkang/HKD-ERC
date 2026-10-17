import torch
import torch.nn as nn


class VUEFusion(nn.Module):
    """
    Fuses a/t/v into one consensus node per utterance, by attention over a
    shared projection, gated by the same log-reliability trick as
    ContextGraphLayer:

        z^m     = W_c h^m
        s^m     = q_c^T tanh(W_s z^m) [+ log w_R(m), if use_reliability_gate]
        gamma   = softmax_m(s^m)
        v_hat   = sum_m gamma^m z^m

    w_R is 1 where the modality is present and a learned reliability where
    it is a placeholder, so with nothing missing (the teacher) the gate
    contributes 0 and this is plain attention fusion. q_c/W_s and the
    reliability parameters are shared across modalities.
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
        h_dict            {'a','t','v': [T, B, concat_dim]}
        availability_dict {'a','t','v': [T, B]}, 1 = present, 0 = placeholder
        returns           v_hat [T, B, d_e], gamma [T, B, M] (per-modality weight)
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
