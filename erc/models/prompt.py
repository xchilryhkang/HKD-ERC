import torch
import torch.nn as nn


class MissingModalityPrompt(nn.Module):
    """
    Simplest placeholder: one learnable constant vector per modality,
    context-independent (doesn't look at what the other modalities say at
    that utterance). Kept as the "constant" prompt_type for ablation against
    ConditionalPrototypePrompt below.
    """

    def __init__(self, dim):
        super(MissingModalityPrompt, self).__init__()
        self.prompt = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.prompt, std=0.02)

    def forward(self, feat, mask):
        """feat [T, B, dim] (already projected to hidden_dim), mask [T, B] 1=present."""
        # Missing slots are zeroed in the RAW input, so after the projection
        # they hold proj(0) = bias, not 0 -- hence the explicit `feat * mask`
        # to discard them rather than relying on them already being zero.
        mask = mask.unsqueeze(-1).to(feat.dtype)  # [T, B, 1]
        return feat * mask + self.prompt.expand_as(feat) * (1 - mask)


class ConditionalPrototypePrompt(nn.Module):
    """
    Conditional prototype prompt (adapted from CPD-GNN): the placeholder for
    a missing modality is an attention-weighted mix of that modality's
    learnable prototype bank, queried by whichever modalities ARE present at
    the same utterance.

        q_i        = mean over present m' of embed(h_i^{m'})
        beta_{i,k} = softmax_k( (W_q q_i)^T (W_p p_k^m) / sqrt(d) )
        placeholder_i^m = unembed_m( sum_k beta_{i,k} p_k^m )

    utils.random_mask guarantees at least one modality survives, so q_i is
    always well-defined.
    """

    def __init__(self, hidden_dim, prompt_dim, n_proto=10, modal_keys=('a', 't', 'v')):
        super(ConditionalPrototypePrompt, self).__init__()
        self.modal_keys = list(modal_keys)
        self.d = prompt_dim

        self.embed = nn.Linear(hidden_dim, prompt_dim)
        self.unembed = nn.ModuleDict({m: nn.Linear(prompt_dim, hidden_dim) for m in self.modal_keys})
        self.proto = nn.ParameterDict({
            m: nn.Parameter(torch.randn(n_proto, prompt_dim) * 0.02) for m in self.modal_keys
        })

        self.w_q = nn.Linear(prompt_dim, prompt_dim, bias=False)
        self.w_p = nn.Linear(prompt_dim, prompt_dim, bias=False)

    def forward(self, feats, mask):
        """
        feats: dict {'a'/'t'/'v': [T, B, hidden_dim]} — already projected,
               zeroed at missing positions upstream.
        mask:  [T, B, len(modal_keys)] — column order matches self.modal_keys,
               1 = present, 0 = missing.
        returns: dict of the same shape as feats, missing slots replaced.
        """
        mask = mask.to(feats[self.modal_keys[0]].dtype)  # [T, B, M]

        # ---- q_i: mean embedding of present modalities only ---- #
        embedded_sum = 0
        for i, m in enumerate(self.modal_keys):
            embedded_sum = embedded_sum + self.embed(feats[m]) * mask[:, :, i:i + 1]
        present_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [T, B, 1]
        q_proj = self.w_q(embedded_sum / present_count)  # [T, B, d]

        out = {}
        for i, m in enumerate(self.modal_keys):
            proto_m = self.proto[m]              # [K, d]
            proto_proj = self.w_p(proto_m)        # [K, d]

            score = torch.einsum('tbd,kd->tbk', q_proj, proto_proj) / (self.d ** 0.5)
            beta = torch.softmax(score, dim=-1)                       # [T, B, K]
            retrieved = torch.einsum('tbk,kd->tbd', beta, proto_m)    # [T, B, d]
            placeholder = self.unembed[m](retrieved)                  # [T, B, hidden_dim]

            m_mask = mask[:, :, i:i + 1]
            out[m] = feats[m] * m_mask + placeholder * (1 - m_mask)
        return out
