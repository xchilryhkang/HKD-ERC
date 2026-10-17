"""
Knowledge-distillation losses between a full-modality teacher
(model.HypergraphERCModel run with mask_rate=0.0, prompt_type='none') and
an incomplete-modality student (same architecture, mask_rate>0,
prompt_type='prototype'/'constant'/'none').

Signals implemented so far:
  - MaskedKLLoss: response-level, KL(student || teacher) on log_prob.
  - PrototypeRelationLoss: relational — match similarity-to-class-prototype
    patterns rather than absolute feature positions. Used both on r =
    concat([a,t,v,e]) and, separately, on the 'e' (VUE/consensus) node
    alone, so 'e' gets its own sharp prototype-relational gradient instead
    of only an indirect one through the concatenated vector.
  - SinkhornOTLoss (+ compute_ot_cost_matrix): optimal transport between the
    two class-probability distributions, with a ground cost that makes
    confusing a class for a semantically NEAR one (happy<->excited) cheap
    and a FAR one (happy<->sad) expensive — "near/far" taken from the
    teacher prototype geometry PrototypeRelationLoss already computes.
  - NodeRelationLoss: structural — match how the 'e' node relates to each
    modality node (a,t,v) AFTER hypergraph refinement, within each model's
    own feature space (teacher's own relation pattern vs student's own).
    This is the hypergraph-node analogue of PrototypeRelationLoss: instead
    of class prototypes, the "reference points" are the model's own other
    nodes for that same utterance.

Still-open candidate for later, per hypergraph.py's HypergraphConvLayer
docstring: gamma^hg (the hyperedge attention weights) as a distillation
target in its own right, beyond the node outputs NodeRelationLoss already
uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedKLLoss(nn.Module):
    """
    Response-level KD: KL(student || teacher) on the classification head,
    with temperature scaling, masked to valid (non-padded) utterances.

    log_prob_student, log_prob_teacher: [T, B, C] — model.py's `log_prob`
    output (already log_softmax'd internally, NOT raw logits). Re-softening
    via log_softmax(log_prob / T) is therefore an approximation — it treats
    log_prob as a logits proxy, which is the common workaround when the
    model only exposes log-probabilities rather than raw logits. If more
    precise response-KD is needed later, the cleanest fix is to have
    HypergraphERCModel.forward also return the pre-log_softmax logits
    (a small addition to model.py's return tuple — not done here since it
    would ripple through every tune_*.py script's unpacking).

    umask: [B, T]
    """

    def __init__(self, temperature=2.0):
        super(MaskedKLLoss, self).__init__()
        self.temperature = temperature
        self.kl = nn.KLDivLoss(reduction='none')

    def forward(self, log_prob_student, log_prob_teacher, umask):
        T = self.temperature

        student_soft = F.log_softmax(log_prob_student / T, dim=-1)
        teacher_soft = F.softmax(log_prob_teacher / T, dim=-1)

        # KLDivLoss(reduction='none') expects input=log-prob, target=prob.
        kl = self.kl(student_soft, teacher_soft).sum(dim=-1)  # [T, B]

        umask_tb = umask.transpose(0, 1)  # [T, B]
        kl = kl * umask_tb
        loss = kl.sum() / umask_tb.sum().clamp(min=1)
        return loss * (T ** 2)


class PrototypeRelationLoss(nn.Module):
    """
    Relational prototype KD: instead of pulling student features onto the
    teacher's absolute positions (fragile when the two spaces differ in
    scale), match the RELATIVE structure — how each valid utterance is
    similarity-distributed around each class prototype.

    Per batch (valid utterances only — padding already removed by caller):
      present   = classes that appear in the batch
      p_s[c]    = mean of student features r_s with label c   (recomputed per batch, has grad)
      p_t[c]    = teacher prototype for class c               (fixed, precomputed per fold)
      S_s       = cos(r_s, p_s[present])  -> [|present|, N]
      S_t       = cos(r_t, p_t[present])  -> [|present|, N]
      loss      = mean((S_s - S_t)^2)     (mean over all |present| x N entries)

    Using the SAME `present` set for both models keeps the two graphs'
    rows aligned; absent classes simply contribute nothing for that batch.

    r_s, r_t: [N, D] — teacher side must be detached (no_grad upstream)
    labels:   [N]
    proto_t:  [K, D] — teacher prototypes, indexed by class id
    """

    def forward(self, r_s, r_t, labels, proto_t):
        if labels.numel() == 0:
            return r_s.sum() * 0.0

        present = torch.unique(labels)

        p_s = torch.stack([r_s[labels == c].mean(dim=0) for c in present])  # [P, D]
        p_t = proto_t[present]                                               # [P, D]

        S_s = F.normalize(p_s, dim=-1) @ F.normalize(r_s, dim=-1).t()       # [P, N]
        S_t = F.normalize(p_t, dim=-1) @ F.normalize(r_t, dim=-1).t()       # [P, N]

        return ((S_s - S_t) ** 2).mean()


def compute_ot_cost_matrix(proto_t):
    """
    Ground cost C for SinkhornOTLoss: Euclidean distance between teacher
    prototypes, AFTER L2-normalizing them (so this is on the same footing
    as PrototypeRelationLoss's cosine — ||a_hat - b_hat||^2 = 2 - 2*cos(a,b)
    for unit vectors — rather than raw-magnitude distance, which the
    anisotropy already observed in this feature space would otherwise
    distort), then scaled to [0, 1] by the largest pairwise distance.

    proto_t: [K, D] teacher prototypes (same ones PrototypeRelationLoss uses).
    returns: [K, K], C[k,k] = 0, C[k,l] in (0, 1] for k != l.
    """
    p_hat = F.normalize(proto_t, dim=-1)
    D = torch.cdist(p_hat, p_hat, p=2)
    return D / D.max().clamp(min=1e-8)


class SinkhornOTLoss(nn.Module):
    """
    Optimal-transport KD: instead of matching the teacher's and student's
    class-probability distributions independently per class (MaskedKLLoss),
    measure the cost of "pouring" P_i into Q_i under a ground cost C that
    makes confusing semantically close classes cheap and distant ones
    expensive (see compute_ot_cost_matrix). Computed per utterance, over
    the full K classes (every class always has nonzero softmax mass, so —
    unlike PrototypeRelationLoss — there is no present/absent-class case
    to handle here).

    Entropic-regularized (Sinkhorn) OT, batched over all N valid utterances
    at once: C (and its kernel exp(-C/eps)) is the SAME [K, K] matrix for
    every utterance (it only depends on the fixed teacher prototypes), so
    only u, v [N, K] vary per utterance — no loop over utterances needed.

        K      = exp(-C / epsilon)                         [K, K]
        u, v   = ones [N, K]
        repeat n_iters times:
            u = P / (v @ K^T)         (row marginal)
            v = Q / (u @ K)           (column marginal)
        gamma_i = diag(u_i) K diag(v_i)                     [K, K] per utterance
        loss_i  = sum(gamma_i * C)
        L_OT    = mean_i(loss_i)

    P, Q: [N, K] probability distributions (rows sum to 1). P is the
    teacher's (must already be detached — it is, since it comes from a
    no_grad forward pass upstream); Q is the student's, with gradient.
    """

    def __init__(self, epsilon=0.1, n_iters=50):
        super(SinkhornOTLoss, self).__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, P, Q, C):
        K_mat = torch.exp(-C / self.epsilon)  # [K, K]

        u = torch.ones_like(P)
        v = torch.ones_like(Q)
        for _ in range(self.n_iters):
            u = P / (v @ K_mat.t()).clamp(min=1e-8)
            v = Q / (u @ K_mat).clamp(min=1e-8)

        gamma = u.unsqueeze(-1) * K_mat.unsqueeze(0) * v.unsqueeze(-2)  # [N, K, K]
        loss_per_utt = (gamma * C.unsqueeze(0)).sum(dim=(-1, -2))        # [N]
        return loss_per_utt.mean()


class NodeRelationLoss(nn.Module):
    """
    Structural KD on the hypergraph's own node graph: match how the 'e'
    (VUE/consensus) node relates to each modality node (a, t, v) AFTER
    hypergraph refinement, between teacher and student — WITHIN each
    model's own feature space rather than across the two models directly.

    Teacher and student are independently-trained networks, so their raw
    h_final vectors live in different coordinate systems even when the
    RELATIONAL structure they encode is nearly identical (confirmed via
    analyze_distribution.py's tsne_overlap_per_modality.png: teacher and
    student occupy fully separate regions per modality, yet
    proto_structure_corr between them is ~0.99). Matching absolute
    positions would fight that mismatch for no benefit; matching each
    model's own internal e-vs-modality relation pattern does not.

    Per valid utterance i, within one model:
        S_i = [cos(e_i, a_i), cos(e_i, t_i), cos(e_i, v_i)]   (3,)

    loss = mean((S_student - S_teacher) ** 2) over all valid utterances.

    h_final_s, h_final_t: dict {'a','t','v','e': [N, D]} — already
    flattened to valid utterances by the caller. The teacher side must
    come from a no_grad forward pass upstream.
    """

    def forward(self, h_final_s, h_final_t):
        if h_final_s['e'].numel() == 0:
            return h_final_s['e'].sum() * 0.0

        def relation(h):
            e = F.normalize(h['e'], dim=-1)
            return torch.stack(
                [(e * F.normalize(h[m], dim=-1)).sum(-1) for m in ('a', 't', 'v')],
                dim=-1
            )  # [N, 3]

        S_s = relation(h_final_s)
        S_t = relation(h_final_t)
        return ((S_s - S_t) ** 2).mean()
