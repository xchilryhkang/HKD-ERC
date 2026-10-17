"""
Isolated analysis: compare the teacher (full modality) and the student
(missing modality, trained with relational prototype KD) on the TEST split
of each fold. Read-only: loads checkpoints, never writes to them, and nothing
imports this file, so it can be deleted without side effects.

Outputs (per fold, under output/analysis/<timestamp>_<dataset>/fold<i>/):
  - metrics.txt                 all scalar metrics
  - tsne.png                    joint t-SNE of concatenated hypergraph features
  - tsne_per_modality.png       joint t-SNE per modality (a/t/v), teacher vs
                                 student, using each modality's own final
                                 hypergraph node (h_final['a'/'t'/'v']) rather
                                 than the concatenated r — lets you see whether
                                 one modality's distribution degrades more
                                 than the others under missing-modality masking,
                                 colored by CLASS (same scheme as tsne.png)
  - tsne_overlap_per_modality.png  same per-modality features, one panel per
                                 modality, teacher/student overlaid in the
                                 same embedding and colored by SOURCE (2
                                 colors) instead of class — shows how much the
                                 student's distribution has drifted/noised
                                 away from the teacher's, per modality
  - prototypes.png              teacher vs student prototype cosine structure
  - confusion.png               row-normalized confusion matrices
  - confidence.png              max-prob and JS-divergence distributions

Also, if --folds has more than one fold, under output/analysis/<timestamp>_<dataset>/:
  - confidence_all_folds.png    same as confidence.png, pooled over all
                                 requested folds' test utterances instead
                                 of just one fold
  - summary_mean.txt            mean/std of every scalar metric over folds

Prototypes here are TEST-set class means for both models (analysis only).
The relational MSE is the held-out analogue of the training L_proto, using
test prototypes on both sides instead of the batch/teacher-train split.
"""

import os
import argparse
import datetime
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import path
from utils import get_loaders, build_model, random_mask, generate_inputs, load_teacher_checkpoint, set_seed
from train_student import concat_repr

EPS = 1e-12


def prepare_batch(data, args, mask_rate, dev):
    audio_host, text_host, visual_host = data[0], data[1], data[2]
    audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
    speaker_info, umask, label = data[6], data[7], data[8]

    if args.dataset == 'MELD':
        bs, sl = speaker_info.shape[0], speaker_info.shape[1]
        qmask = torch.zeros(bs, sl, dtype=torch.long)
        qmask_9 = torch.argmax(speaker_info, dim=2).long()
    else:
        qmask = speaker_info

    adim, tdim, vdim = audio_host.size(2), text_host.size(2), visual_host.size(2)
    seqlen, batch = audio_host.size(0), audio_host.size(1)

    matrix = random_mask(3, seqlen * batch, mask_rate)
    a_h = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
    t_h = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
    v_h = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))
    matrix = random_mask(3, seqlen * batch, mask_rate)
    a_g = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
    t_g = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
    v_g = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))

    to = lambda x: x.to(dev)
    audio_host, text_host, visual_host = to(audio_host), to(text_host), to(visual_host)
    audio_guest, text_guest, visual_guest = to(audio_guest), to(text_guest), to(visual_guest)
    a_h, t_h, v_h, a_g, t_g, v_g = to(a_h), to(t_h), to(v_h), to(a_g), to(t_g), to(v_g)
    qmask, umask, label = to(qmask), to(umask), to(label)
    model_qmask = to(qmask_9) if args.dataset == 'MELD' else qmask

    lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

    student_feats = generate_inputs(
        audio_host * a_h, text_host * t_h, visual_host * v_h,
        audio_guest * a_g, text_guest * t_g, visual_guest * v_g, qmask)
    student_mask = generate_inputs(a_h, t_h, v_h, a_g, t_g, v_g, qmask)
    teacher_feats = generate_inputs(audio_host, text_host, visual_host,
                                    audio_guest, text_guest, visual_guest, qmask)
    teacher_mask = [torch.ones_like(student_mask[0])]

    return dict(student_feats=student_feats, student_mask=student_mask,
                teacher_feats=teacher_feats, teacher_mask=teacher_mask,
                model_qmask=model_qmask, umask=umask, label=label,
                lengths=lengths, adim=adim, tdim=tdim, vdim=vdim)


@torch.no_grad()
def collect(args, teacher, student, loader, mask_rate, dev):
    """Run both models over a loader; return valid-utterance arrays."""
    teacher.eval()
    student.eval()
    keys = ['label', 'pt', 'ps', 'rt', 'rs',
             'rt_a', 'rt_t', 'rt_v', 'rs_a', 'rs_t', 'rs_v']
    acc = {k: [] for k in keys}

    for data in loader:
        b = prepare_batch(data, args, mask_rate, dev)
        lp_s, _, _, hs, _ = student(b['student_feats'], b['model_qmask'], b['umask'], b['lengths'],
                                    b['student_mask'], b['adim'], b['tdim'], b['vdim'], b['label'])
        lp_t, _, _, ht, _ = teacher(b['teacher_feats'], b['model_qmask'], b['umask'], b['lengths'],
                                    b['teacher_mask'], b['adim'], b['tdim'], b['vdim'], b['label'])

        valid = b['umask'].reshape(-1).bool()
        C = lp_s.size(-1)
        D = concat_repr(hs).size(-1)

        acc['label'].append(b['label'].reshape(-1)[valid].cpu().numpy())
        acc['pt'].append(lp_t.transpose(0, 1).reshape(-1, C)[valid].exp().cpu().numpy())
        acc['ps'].append(lp_s.transpose(0, 1).reshape(-1, C)[valid].exp().cpu().numpy())
        acc['rt'].append(concat_repr(ht).transpose(0, 1).reshape(-1, D)[valid].cpu().numpy())
        acc['rs'].append(concat_repr(hs).transpose(0, 1).reshape(-1, D)[valid].cpu().numpy())

        for m in ('a', 't', 'v'):
            Dm = ht[m].size(-1)
            acc[f'rt_{m}'].append(ht[m].transpose(0, 1).reshape(-1, Dm)[valid].cpu().numpy())
            acc[f'rs_{m}'].append(hs[m].transpose(0, 1).reshape(-1, Dm)[valid].cpu().numpy())

    return {k: np.concatenate(v) for k, v in acc.items()}


def cosine_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + EPS)
    return A @ B.T


def compute_metrics(d, n_classes):
    lab, pt, ps, rt, rs = d['label'], d['pt'], d['ps'], d['rt'], d['rs']
    N = len(lab)
    pred_t, pred_s = pt.argmax(1), ps.argmax(1)
    classes = np.unique(lab)

    m = {}
    m['N_valid_utterances'] = N
    m['acc_teacher'] = accuracy_score(lab, pred_t)
    m['acc_student'] = accuracy_score(lab, pred_s)
    m['f1_teacher'] = f1_score(lab, pred_t, average='weighted')
    m['f1_student'] = f1_score(lab, pred_s, average='weighted')
    m['pred_agreement'] = float((pred_t == pred_s).mean())

    # --- distribution of class probabilities ---
    M = 0.5 * (pt + ps)
    kl_tm = (pt * (np.log(pt + EPS) - np.log(M + EPS))).sum(1)
    kl_sm = (ps * (np.log(ps + EPS) - np.log(M + EPS))).sum(1)
    js = 0.5 * (kl_tm + kl_sm)
    m['js_divergence_mean'] = float(js.mean())
    m['confidence_teacher'] = float(pt.max(1).mean())
    m['confidence_student'] = float(ps.max(1).mean())
    m['p_true_teacher'] = float(pt[np.arange(N), lab].mean())
    m['p_true_student'] = float(ps[np.arange(N), lab].mean())

    # --- prototypes (test-set class means, same for both sides) ---
    mu_t = np.stack([rt[lab == c].mean(0) for c in classes])
    mu_s = np.stack([rs[lab == c].mean(0) for c in classes])
    per_class_cos = np.diag(cosine_matrix(mu_t, mu_s))
    m['proto_cos_mean'] = float(per_class_cos.mean())
    m['proto_cos_min'] = float(per_class_cos.min())

    Ct = cosine_matrix(mu_t, mu_t)
    Cs = cosine_matrix(mu_s, mu_s)
    if len(classes) > 2:
        iu = np.triu_indices(len(classes), 1)
        m['proto_structure_corr'] = float(np.corrcoef(Ct[iu], Cs[iu])[0, 1])
    else:
        m['proto_structure_corr'] = float('nan')

    # --- relational MSE on test (held-out analogue of L_proto) ---
    S_t = cosine_matrix(rt, mu_t)
    S_s = cosine_matrix(rs, mu_s)
    m['relational_mse'] = float(((S_t - S_s) ** 2).mean())

    # --- feature geometry ---
    rng = np.random.RandomState(0)
    sub = rng.choice(N, size=min(2000, N), replace=False)
    m['silhouette_teacher'] = float(silhouette_score(
        rt[sub] / (np.linalg.norm(rt[sub], axis=1, keepdims=True) + EPS), lab[sub]))
    m['silhouette_student'] = float(silhouette_score(
        rs[sub] / (np.linalg.norm(rs[sub], axis=1, keepdims=True) + EPS), lab[sub]))

    # --- per-class F1 ---
    f1_t = f1_score(lab, pred_t, average=None, labels=range(n_classes), zero_division=0)
    f1_s = f1_score(lab, pred_s, average=None, labels=range(n_classes), zero_division=0)
    for c in range(n_classes):
        m[f'f1_class{c}_teacher'] = float(f1_t[c])
        m[f'f1_class{c}_student'] = float(f1_s[c])

    arrays = dict(js=js, Ct=Ct, Cs=Cs, per_class_cos=per_class_cos, classes=classes,
                  pred_t=pred_t, pred_s=pred_s, lab=lab, pt=pt, ps=ps)
    return m, arrays


def plot_tsne(d, out_path, max_points=1500, seed=0):
    rng = np.random.RandomState(seed)
    n = len(d['label'])
    idx = rng.choice(n, size=min(max_points, n), replace=False)

    rt = d['rt'][idx] / (np.linalg.norm(d['rt'][idx], axis=1, keepdims=True) + EPS)
    rs = d['rs'][idx] / (np.linalg.norm(d['rs'][idx], axis=1, keepdims=True) + EPS)
    joint = np.concatenate([rt, rs], axis=0)
    emb = TSNE(n_components=2, perplexity=min(30, len(joint) // 4), init='pca',
               random_state=seed).fit_transform(joint)
    et, es = emb[:len(idx)], emb[len(idx):]
    lab = d['label'][idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    cmap = plt.get_cmap('tab10')
    for ax, e, title in [(axes[0], et, 'Teacher (full modality)'), (axes[1], es, 'Student (missing modality)')]:
        for c in np.unique(lab):
            mask = lab == c
            ax.scatter(e[mask, 0], e[mask, 1], s=8, alpha=0.6, color=cmap(c % 10), label=f'class {c}')
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(loc='best', fontsize=8, markerscale=2)
    fig.suptitle('t-SNE of concatenated hypergraph features (shared embedding)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_tsne_per_modality(d, out_path, max_points=1500, seed=0):
    """
    Same idea as plot_tsne, but one row per modality (a/t/v), using each
    modality's own final hypergraph node h_final['a'/'t'/'v'] instead of the
    concatenated r. Each row is its own joint t-SNE fit (teacher+student of
    THAT modality only) — rows are not comparable to each other spatially,
    only teacher-vs-student within the same row is.
    """
    rng = np.random.RandomState(seed)
    n = len(d['label'])
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    lab = d['label'][idx]
    modalities = [('a', 'Audio'), ('t', 'Text'), ('v', 'Visual')]

    fig, axes = plt.subplots(3, 2, figsize=(12, 15.5))
    cmap = plt.get_cmap('tab10')

    for row, (mkey, mname) in enumerate(modalities):
        rt = d[f'rt_{mkey}'][idx]
        rs = d[f'rs_{mkey}'][idx]
        rt = rt / (np.linalg.norm(rt, axis=1, keepdims=True) + EPS)
        rs = rs / (np.linalg.norm(rs, axis=1, keepdims=True) + EPS)
        joint = np.concatenate([rt, rs], axis=0)
        emb = TSNE(n_components=2, perplexity=min(30, len(joint) // 4), init='pca',
                   random_state=seed).fit_transform(joint)
        et, es = emb[:len(idx)], emb[len(idx):]

        for col, (e, title) in enumerate([(et, f'{mname} — Teacher'), (es, f'{mname} — Student')]):
            ax = axes[row, col]
            for c in np.unique(lab):
                mask = lab == c
                ax.scatter(e[mask, 0], e[mask, 1], s=8, alpha=0.6, color=cmap(c % 10), label=f'class {c}')
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])

    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, markerscale=2)
    fig.suptitle('t-SNE per modality (final hypergraph node), shared embedding per row')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_tsne_overlap_per_modality(d, out_path, max_points=1500, seed=0):
    """
    One figure, one panel per modality (a/t/v): teacher and student points
    overlaid in the SAME embedding, colored by SOURCE (teacher vs student)
    rather than by class. This is for inspecting how much the student's
    per-modality feature distribution has drifted/noised relative to the
    teacher's under missing-modality masking — heavy overlap means the
    student's features for that modality stay close to the teacher's
    (low noise); a clean split into two separate blobs means the student's
    distribution for that modality has shifted away from the teacher's
    (high noise/drift), independent of whether classes are separable.
    """
    rng = np.random.RandomState(seed)
    n = len(d['label'])
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    modalities = [('a', 'Audio'), ('t', 'Text'), ('v', 'Visual')]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for col, (mkey, mname) in enumerate(modalities):
        rt = d[f'rt_{mkey}'][idx]
        rs = d[f'rs_{mkey}'][idx]
        rt = rt / (np.linalg.norm(rt, axis=1, keepdims=True) + EPS)
        rs = rs / (np.linalg.norm(rs, axis=1, keepdims=True) + EPS)
        joint = np.concatenate([rt, rs], axis=0)
        emb = TSNE(n_components=2, perplexity=min(30, len(joint) // 4), init='pca',
                   random_state=seed).fit_transform(joint)
        et, es = emb[:len(idx)], emb[len(idx):]

        ax = axes[col]
        ax.scatter(et[:, 0], et[:, 1], s=8, alpha=0.5, color='tab:blue', label='Teacher')
        ax.scatter(es[:, 0], es[:, 1], s=8, alpha=0.5, color='tab:orange', label='Student')
        ax.set_title(mname)
        ax.set_xticks([]); ax.set_yticks([])

    axes[-1].legend(loc='best', fontsize=9, markerscale=2)
    fig.suptitle('t-SNE per modality, colored by source (teacher vs student) — overlap = low drift/noise')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_prototypes(arr, out_path):
    classes = arr['classes']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, M, title in [(axes[0], arr['Ct'], 'Teacher prototype cosine'),
                         (axes[1], arr['Cs'], 'Student prototype cosine')]:
        im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_title(title)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[2].bar([str(c) for c in classes], arr['per_class_cos'], color='gray')
    axes[2].set_ylim(-1, 1)
    axes[2].set_title('cos(teacher proto, student proto) per class')
    axes[2].set_xlabel('class')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(arr, n_classes, out_path):
    lab = arr['lab']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, pred, title in [(axes[0], arr['pred_t'], 'Teacher'), (axes[1], arr['pred_s'], 'Student')]:
        cm = confusion_matrix(lab, pred, labels=range(n_classes)).astype(float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cm, vmin=0, vmax=1, cmap='Blues')
        ax.set_title(f'{title} (row-normalized)')
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > 0.5 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


_DRIFT_COLOR_FULL = '#0072B2'   # blue (Okabe-Ito, colorblind-safe)
_DRIFT_COLOR_MISS = '#E69F00'   # orange
_DRIFT_COLOR_JS = '#555555'


def plot_confidence(d, arr, out_path, suptitle=None, mask_rate=None, bins=30):
    """
    mask_rate/suptitle are optional -- when given, matches the visual
    convention of an earlier "drift_<dataset>_..._fold<i>" analysis
    (full-modality model vs missing-modality model, i.e. teacher vs
    student here): same colors/figsize/dpi/shared-bin-edges/axis labels as
    that script's plot(), so output is directly comparable to its old
    plots. Without mask_rate, falls back to the original teacher/student
    framing with matplotlib's default styling.

    argmax_changed is the fraction of valid utterances where the two
    models' predicted class differs (1 - pred_agreement, as a percentage).
    """
    pred_t, pred_s = arr['pred_t'], arr['pred_s']
    conf_full, conf_miss = arr['pt'].max(1), arr['ps'].max(1)
    argmax_changed = float((pred_t != pred_s).mean())
    js = arr['js']
    js_mean, js_median = float(np.mean(js)), float(np.median(js))

    if mask_rate is not None:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6))
        edges = np.linspace(min(conf_full.min(), conf_miss.min()), 1.0, bins + 1)
        a0.hist(conf_full, bins=edges, density=True, alpha=0.6, color=_DRIFT_COLOR_FULL, label='full (mr 0)')
        a0.hist(conf_miss, bins=edges, density=True, alpha=0.6, color=_DRIFT_COLOR_MISS, label=f'missing (mr {mask_rate:g})')
        a0.set_title('max class probability (confidence)')
        a0.set_xlabel('max probability')
        a0.legend(loc='upper left')

        a1.hist(js, bins=bins, density=True, color=_DRIFT_COLOR_JS)
        a1.axvline(js_mean, color='k', ls='--', lw=1)
        a1.set_title(f'per-utterance JS divergence (full vs missing)\n'
                     f'mean {js_mean:.4f}, median {js_median:.4f}, argmax changed {argmax_changed:.1%}')
        a1.set_xlabel('JS divergence (nats)')

        if suptitle:
            fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(conf_full, bins=30, alpha=0.6, label='teacher', density=True)
    axes[0].hist(conf_miss, bins=30, alpha=0.6, label='student', density=True)
    axes[0].set_title('max class probability (confidence)')
    axes[0].legend()
    axes[1].hist(js, bins=30, color='gray', density=True)
    axes[1].set_title('per-utterance JS divergence (teacher vs student)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IEMOCAPSix')
    parser.add_argument('--folds', type=str, default='0', help='Comma-separated fold indices, e.g. "0,1,2,3,4".')
    parser.add_argument('--mask-rate', type=float, default=0.2,
                        help='Missing rate applied to the student at test time (should match the student run).')
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=50)

    # architecture: must match the teacher/student checkpoints
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--d-e', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)
    parser.add_argument('--prompt-type', type=str, default='prototype', choices=['none', 'constant', 'prototype'],
                        help='Must match the prompt_type the student checkpoint being loaded was trained with.')
    args = parser.parse_args()

    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.mask_type = f'constant-{args.mask_rate}'

    if args.dataset == 'IEMOCAPFour':
        args.num_folder, args.n_classes, args.n_speakers = 5, 4, 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder, args.n_classes, args.n_speakers = 5, 6, 2
    elif args.dataset == 'MELD':
        args.num_folder, args.n_classes, args.n_speakers = 1, 7, 9
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    set_seed(args.seed, dev.type == 'cuda')
    args.device = dev
    args.no_cuda = dev.type == 'cpu'

    if args.dataset == 'MELD':
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=0, meld_path=path.PATH_MELD)
    else:
        audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
        text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
        video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            audio_root=audio_root, text_root=text_root, video_root=video_root,
            num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0)

    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    base_dir = os.path.join(path.SAVED_ROOT, 'analysis', f'{stamp}_{args.dataset}')
    os.makedirs(base_dir, exist_ok=True)

    folds = [int(x) for x in args.folds.split(',')]
    all_metrics = []
    all_pt, all_ps, all_js, all_pred_t, all_pred_s = [], [], [], [], []

    for fold in folds:
        print(f'\n===== fold {fold} =====')
        fold_dir = os.path.join(base_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        teacher_args = copy.deepcopy(args)
        teacher_args.prompt_type = 'none'
        teacher = build_model(teacher_args, adim, tdim, vdim).to(dev)
        teacher.load_state_dict(load_teacher_checkpoint(args.dataset, fold_index=fold, map_location=dev))

        student = build_model(args, adim, tdim, vdim).to(dev)
        student_ckpt = os.path.join(path.STUDENT_DIR, f'{args.dataset.lower()}_fold{fold}_student.pt')
        student.load_state_dict(torch.load(student_ckpt, map_location=dev))

        d = collect(args, teacher, student, test_loaders[fold], args.mask_rate, dev)
        m, arr = compute_metrics(d, args.n_classes)
        all_metrics.append(m)

        with open(os.path.join(fold_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
            for k, v in m.items():
                line = f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}'
                f.write(line + '\n')
                print(line)

        plot_tsne(d, os.path.join(fold_dir, 'tsne.png'))
        plot_tsne_per_modality(d, os.path.join(fold_dir, 'tsne_per_modality.png'))
        plot_tsne_overlap_per_modality(d, os.path.join(fold_dir, 'tsne_overlap_per_modality.png'))
        plot_prototypes(arr, os.path.join(fold_dir, 'prototypes.png'))
        plot_confusion(arr, args.n_classes, os.path.join(fold_dir, 'confusion.png'))
        plot_confidence(d, arr, os.path.join(fold_dir, 'confidence.png'))
        drift_name = f'drift_{args.dataset}_mr{args.mask_rate:g}_fold{fold + 1}'
        plot_confidence(
            d, arr, os.path.join(fold_dir, f'{drift_name}.svg'),
            suptitle=drift_name, mask_rate=args.mask_rate
        )

        all_pt.append(arr['pt'])
        all_ps.append(arr['ps'])
        all_js.append(arr['js'])
        all_pred_t.append(arr['pred_t'])
        all_pred_s.append(arr['pred_s'])

    if len(folds) > 1:
        arr_all = dict(pt=np.concatenate(all_pt), ps=np.concatenate(all_ps), js=np.concatenate(all_js),
                        pred_t=np.concatenate(all_pred_t), pred_s=np.concatenate(all_pred_s))
        plot_confidence(None, arr_all, os.path.join(base_dir, 'confidence_all_folds.png'))
        pooled_drift_name = f'drift_{args.dataset}_mr{args.mask_rate:g}_pooled'
        plot_confidence(
            None, arr_all, os.path.join(base_dir, f'{pooled_drift_name}.svg'),
            suptitle=f'{pooled_drift_name} (all folds, n={len(arr_all["js"])})', mask_rate=args.mask_rate
        )
        print(f'\nconfidence_all_folds.png + {pooled_drift_name}.svg written '
              f'(N={len(arr_all["js"])} utterances pooled over {len(folds)} folds)')

    if len(all_metrics) > 1:
        print('\n===== mean over folds =====')
        summary_path = os.path.join(base_dir, 'summary_mean.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            for k in all_metrics[0]:
                vals = [m[k] for m in all_metrics if isinstance(m[k], float) and not np.isnan(m[k])]
                if vals:
                    line = f'{k}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}'
                    f.write(line + '\n')
                    print(line)

    print(f'\nOutputs written to: {base_dir}')
