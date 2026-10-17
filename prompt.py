import torch
import torch.nn as nn


# ============================ #
#  Missing-modality prompts
# ============================ #
#
# Both prompt modules below operate AFTER each modality has already been
# linearly projected to the shared `hidden_dim` (see HypergraphERCModel.proj_a/t/v
# in model.py) and BEFORE the optional BiLSTM refinement — i.e. exactly at
# the point HypergraphERCModel.forward calls them, so "dim" here always means
# hidden_dim, never the modality's raw adim/tdim/vdim.
#
# Raw input is zeroed at missing positions upstream (train.py:
# masked_audio_host = audio_host * audio_host_mask), so even after the
# Linear projection a missing slot is `proj(0) = bias` (not literally zero).
# Both modules multiply that slot by mask==0 before adding the placeholder,
# so whatever the projector produced there is discarded regardless of its
# value — only genuinely present positions keep their projected feature.


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
        """
        feat: [T, B, dim]
        mask: [T, B] — 1 = present, 0 = missing.
        """
        mask = mask.unsqueeze(-1).to(feat.dtype)  # [T, B, 1]
        return feat * mask + self.prompt.expand_as(feat) * (1 - mask)


class ConditionalPrototypePrompt(nn.Module):
    """
    Conditional prototype prompt (adapted from CPD-GNN). Each modality m has
    a learnable prototype bank P^m in R^{K x d}. The placeholder for a
    missing modality m at utterance i is an attention-weighted combination
    of P^m, conditioned on a query built from whichever modalities ARE
    present at that same utterance:

        q_i        = mean over present m' of embed(h_i^{m'})
        beta_{i,k} = softmax_k( (W_q q_i)^T (W_p p_k^m) / sqrt(d) )
        h~_i^m     = sum_k beta_{i,k} * p_k^m            (in space d)
        placeholder_i^m = unembed_m(h~_i^m)               (back to hidden_dim)

    h_i^{m'} is the modality's own hidden_dim representation (HypergraphERCModel's
    proj_a/t/v output) — there's always at least one present modality by
    construction of the missing-rate masking (utils.random_mask guarantees
    it), so q_i is always well-defined.

    All three modalities already share `hidden_dim` by the time this module
    runs (see model.py), so there is no per-modality input projection here
    — only a shared `embed`/`unembed` pair into the (possibly different)
    attention space `d` = prompt_dim, plus one prototype bank per modality.
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
