import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedKLLoss(nn.Module):
    """
    Temperature-scaled KL(student || teacher) on the head, masked to valid
    utterances. Inputs are [T, B, C] log_prob (model.py log_softmaxes
    internally), umask is [B, T].

    Re-softening log_prob as if it were logits is an approximation; getting
    it exact would mean returning raw logits from the model too, which would
    ripple through every caller's unpacking.
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
    Matches the RELATIVE structure of each model's own feature space —
    cosine similarity of every utterance to its class prototypes — rather
    than pulling student features onto the teacher's absolute positions,
    which is fragile when the two spaces differ in scale.

    r_s, r_t [N, D] (teacher side detached), labels [N], proto_t [K, D].
    Both sides use the same `present` class set so their rows stay aligned.
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
    Ground cost for SinkhornOTLoss: pairwise distance between teacher
    prototypes [K, D] -> [K, K] in [0, 1].

    Prototypes are L2-normalized first so the cost is cosine-based like
    PrototypeRelationLoss; raw-magnitude distance would be distorted by
    this feature space's anisotropy.
    """
    p_hat = F.normalize(proto_t, dim=-1)
    D = torch.cdist(p_hat, p_hat, p=2)
    return D / D.max().clamp(min=1e-8)


class SinkhornOTLoss(nn.Module):
    """
    Entropic-regularized (Sinkhorn) OT between the teacher's and student's
    class distributions, under a ground cost that makes confusing close
    classes cheap and distant ones expensive — unlike MaskedKLLoss, which
    treats every class as equally far apart.

    P, Q [N, K] are row-stochastic; P is the teacher's (detached upstream).
    C is the same [K, K] for every utterance, so all N are solved at once.
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
    Matches how the 'e' (consensus) node relates to a/t/v after hypergraph
    refinement: per utterance, [cos(e,a), cos(e,t), cos(e,v)] computed
    WITHIN each model, then compared.

    Teacher and student are separately trained, so their h_final vectors sit
    in different coordinate systems even when the structure they encode is
    nearly identical (they occupy disjoint t-SNE regions yet their prototype
    structure correlates ~0.99). Comparing relations rather than absolute
    positions sidesteps that mismatch.

    h_final_s/_t: {'a','t','v','e': [N, D]}, teacher side detached upstream.
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
