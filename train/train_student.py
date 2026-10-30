import os
import sys
import time
import argparse
import datetime
import copy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

# Make the repo root importable so `src` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import path
from src.utils import (
    Logger, get_loaders, build_model, random_mask, generate_inputs,
    load_teacher_checkpoint, save_student_checkpoint, make_optimizer, set_seed,
    regression_metrics, bin_sentiment
)
from src.losses.loss import MaskedCELoss
from src.losses.kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)


def concat_repr(h_final):
    """r = [h_a || h_t || h_v || h_e], the representation the classifier consumes.
    h_final values are [T, B, D] -> result [T, B, 4D]."""
    return torch.cat([h_final['a'], h_final['t'], h_final['v'], h_final['e']], dim=-1)


def flatten_valid(x, valid):
    """[T, B, D] -> [N, D], valid (non-padded) utterances only."""
    return x.transpose(0, 1).reshape(-1, x.size(-1))[valid]


def compute_teacher_prototypes(args, teacher, dataloader, adim, tdim, vdim):
    """
    One pass of the frozen teacher over the whole train split. Returns
    (proto_t, proto_t_e):
      proto_t   [n_classes, 4*hidden_dim] - per-class mean of teacher r (= concat
                of all 4 hypergraph nodes) over valid utterances
      proto_t_e [n_classes, hidden_dim]   - per-class mean of teacher h_final['e']
                alone, same utterances, same pass (no extra forward cost)
    Both fixed for the rest of this fold.
    """
    cuda = torch.cuda.is_available() and not args.no_cuda
    teacher.eval()

    sums, counts = None, None
    sums_e = None

    with torch.no_grad():
        for data in dataloader:
            audio_host, text_host, visual_host = data[0], data[1], data[2]
            audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
            speaker_info, umask, label = data[6], data[7], data[8]

            if args.dataset == 'MELD':
                batch_size, seq_len = speaker_info.shape[0], speaker_info.shape[1]
                qmask = torch.zeros(batch_size, seq_len, dtype=torch.long)
                qmask_9 = torch.argmax(speaker_info, dim=2).long()
            else:
                qmask = speaker_info

            seqlen, batch = audio_host.size(0), audio_host.size(1)

            if cuda:
                audio_host, text_host, visual_host = audio_host.cuda(), text_host.cuda(), visual_host.cuda()
                audio_guest, text_guest, visual_guest = audio_guest.cuda(), text_guest.cuda(), visual_guest.cuda()
                qmask, umask, label = qmask.cuda(), umask.cuda(), label.cuda()
                if args.dataset == 'MELD':
                    qmask_9 = qmask_9.cuda()

            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            model_qmask = qmask_9 if args.dataset == 'MELD' else qmask

            teacher_input_features = generate_inputs(
                audio_host, text_host, visual_host,
                audio_guest, text_guest, visual_guest, qmask)
            teacher_mask_ones = [torch.ones(seqlen, batch, 3, dtype=torch.long,
                                            device=teacher_input_features[0].device)]

            _, _, _, h_final, _ = teacher(
                teacher_input_features, model_qmask, umask, lengths, teacher_mask_ones,
                adim, tdim, vdim, label
            )

            r = concat_repr(h_final)                                     # [T, B, D]
            D = r.size(-1)
            valid = umask.reshape(-1).bool()
            r_v = flatten_valid(r, valid)                                 # [N, D]
            e_v = flatten_valid(h_final['e'], valid)                      # [N, hidden_dim]
            lab_v = label.reshape(-1)[valid]                             # [N]

            if getattr(args, 'task', 'classification') == 'regression':
                # Prototypes are per-class, so the continuous sentiment score
                # is binned into args.proto_bins pseudo-classes -- the same
                # binning train_or_eval_student applies, so the rows line up.
                lab_v, keep = bin_sentiment(lab_v, getattr(args, 'proto_bins', 2),
                                            getattr(args, 'proto_bin_mode', 'uniform'))
                lab_v, r_v, e_v = lab_v[keep], r_v[keep], e_v[keep]

            n_proto_cls = getattr(args, 'n_proto_classes', args.n_classes)
            if sums is None:
                sums = torch.zeros(n_proto_cls, D, device=r.device)
                counts = torch.zeros(n_proto_cls, device=r.device)
                sums_e = torch.zeros(n_proto_cls, e_v.size(-1), device=r.device)
            sums.index_add_(0, lab_v, r_v)
            sums_e.index_add_(0, lab_v, e_v)
            counts.index_add_(0, lab_v, torch.ones_like(lab_v, dtype=counts.dtype))

    if getattr(args, 'task', 'classification') == 'regression':
        # A prototype is a mean over its bin, so thin bins give noisy targets.
        # Print the support so a too-large --proto-bins is visible, not silent.
        n = [int(c) for c in counts]
        print(f"  prototype bin support (train, {len(n)} "
              f"{getattr(args, 'proto_bin_mode', 'uniform')} bins): {n}")
        if min(n) < 100:
            print(f'  WARNING: smallest bin has {min(n)} utterances for a '
                  f'{int(sums.size(1))}-dim prototype -- consider fewer --proto-bins.')

    denom = counts.clamp(min=1).unsqueeze(1)
    return sums / denom, sums_e / denom                                 # [n_proto_cls, D], [n_proto_cls, hidden_dim]


def calibrate_loss_scales(args, *fwd_args, **fwd_kwargs):
    """
    Measure how large proto and rel actually are next to CE, so that a lambda
    of 1 means "weigh this cluster like CE" instead of "weigh it at a fraction
    of a percent". Returns (proto_scale, rel_scale).

    The ratios are dataset-dependent (CE/proto is ~270 on MELD but ~90 on
    IEMOCAP), which is why they are measured per run rather than hardcoded.

    Call this ONCE per run and reuse the result for every lambda: scales that
    moved between settings would make the same lambda mean different things
    in different cells of a sweep.
    """
    # Only the raw per-cluster averages are read, so neutralise the weights
    # (and tolerate a caller that has not set them yet), then put back
    # whatever was there -- including the RNG, which the loader consumes.
    saved = {k: getattr(args, k, None)
             for k in ('lambda_proto', 'lambda_rel', 'proto_scale', 'rel_scale')}
    for k in saved:
        setattr(args, k, 1.0)
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()

    _, _, _, ce, _, proto, _, e_proto, node_rel = train_or_eval_student(
        args, *fwd_args, optimizer=None, train=False, **fwd_kwargs)

    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    for k, v in saved.items():
        if v is None:
            delattr(args, k)
        else:
            setattr(args, k, v)

    rel = e_proto + node_rel
    return ce / max(proto, 1e-8), ce / max(rel, 1e-8)


def train_or_eval_student(args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                           ot_loss, ot_cost, node_rel_loss, dataloader, mask_rate, optimizer=None, train=False, return_extra=False):
    preds, labels, masks = [], [], []
    losses, losses_ce, losses_kd, losses_proto, losses_ot = [], [], [], [], []
    losses_e_proto, losses_node_rel = [], []

    cuda = torch.cuda.is_available() and not args.no_cuda
    teacher.eval()
    student.train() if train else student.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        speaker_info, umask, label = data[6], data[7], data[8]

        if args.dataset == 'MELD':
            batch_size, seq_len = speaker_info.shape[0], speaker_info.shape[1]
            qmask = torch.zeros(batch_size, seq_len, dtype=torch.long)
            qmask_9 = torch.argmax(speaker_info, dim=2).long()
        else:
            qmask = speaker_info

        adim, tdim, vdim = audio_host.size(2), text_host.size(2), visual_host.size(2)
        seqlen, batch = audio_host.size(0), audio_host.size(1)

        matrix = random_mask(3, seqlen * batch, mask_rate)
        audio_host_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
        text_host_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
        visual_host_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))
        matrix = random_mask(3, seqlen * batch, mask_rate)
        audio_guest_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
        text_guest_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
        visual_guest_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        if cuda:
            audio_host, text_host, visual_host = audio_host.cuda(), text_host.cuda(), visual_host.cuda()
            audio_guest, text_guest, visual_guest = audio_guest.cuda(), text_guest.cuda(), visual_guest.cuda()
            masked_audio_host, audio_host_mask = masked_audio_host.cuda(), audio_host_mask.cuda()
            masked_text_host, text_host_mask = masked_text_host.cuda(), text_host_mask.cuda()
            masked_visual_host, visual_host_mask = masked_visual_host.cuda(), visual_host_mask.cuda()
            masked_audio_guest, audio_guest_mask = masked_audio_guest.cuda(), audio_guest_mask.cuda()
            masked_text_guest, text_guest_mask = masked_text_guest.cuda(), text_guest_mask.cuda()
            masked_visual_guest, visual_guest_mask = masked_visual_guest.cuda(), visual_guest_mask.cuda()
            qmask, umask, label = qmask.cuda(), umask.cuda(), label.cuda()
            if args.dataset == 'MELD':
                qmask_9 = qmask_9.cuda()

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        model_qmask = qmask_9 if args.dataset == 'MELD' else qmask

        masked_input_features = generate_inputs(
            masked_audio_host, masked_text_host, masked_visual_host,
            masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(
            audio_host_mask, text_host_mask, visual_host_mask,
            audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        # Teacher sees complete data: all-ones mask, so nothing downstream
        # treats any modality as missing.
        teacher_input_features = generate_inputs(
            audio_host, text_host, visual_host,
            audio_guest, text_guest, visual_guest, qmask)
        teacher_mask_ones = [torch.ones_like(input_features_mask[0])]

        log_prob_student, v_hat_s, gamma_vue_s, h_final_s, gammas_hg_s = student(
            masked_input_features, model_qmask, umask, lengths, input_features_mask,
            adim, tdim, vdim, label
        )

        with torch.no_grad():
            log_prob_teacher, v_hat_t, gamma_vue_t, h_final_t, gammas_hg_t = teacher(
                teacher_input_features, model_qmask, umask, lengths, teacher_mask_ones,
                adim, tdim, vdim, label
            )

        lp_ = log_prob_student.transpose(0, 1).contiguous().view(-1, log_prob_student.size(2))
        labels_ = label.view(-1)

        regression = getattr(args, 'task', 'classification') == 'regression'

        loss_ce = cls_loss(lp_, labels_, umask)
        lp_t_ = log_prob_teacher.transpose(0, 1).contiguous().view(-1, log_prob_teacher.size(2))

        # Flatten the 4 hypergraph nodes to valid utterances once; every
        # feature-level loss below is built from these.
        valid = umask.reshape(-1).bool()
        hf_s_v = {k: flatten_valid(v, valid) for k, v in h_final_s.items()}  # has grad
        hf_t_v = {k: flatten_valid(v, valid) for k, v in h_final_t.items()}  # no grad
        lab_v = labels_[valid]                                        # [N]

        r_s_v = torch.cat([hf_s_v['a'], hf_s_v['t'], hf_s_v['v'], hf_s_v['e']], dim=-1)
        r_t_v = torch.cat([hf_t_v['a'], hf_t_v['t'], hf_t_v['v'], hf_t_v['e']], dim=-1)

        if regression:
            # prob cluster: KL and Sinkhorn OT both need a distribution over
            # classes, which the 1-wide regression head cannot produce. The
            # student is matched to the teacher scalar instead (plain
            # regression distillation) and OT drops out of the sum, so
            # lambda_prob still scales the whole prob cluster.
            loss_kd = cls_loss(lp_, lp_t_.detach(), umask)
            loss_ot = torch.zeros((), device=lp_.device)

            # proto cluster: same binning compute_teacher_prototypes used.
            lab_p, keep = bin_sentiment(lab_v, getattr(args, 'proto_bins', 2),
                                        getattr(args, 'proto_bin_mode', 'uniform'))
            if keep.any():
                lab_p = lab_p[keep]
                loss_proto = proto_loss(r_s_v[keep], r_t_v[keep], lab_p, proto_t)
                loss_e_proto = proto_loss(hf_s_v['e'][keep], hf_t_v['e'][keep], lab_p, proto_t_e)
            else:
                loss_proto = torch.zeros((), device=lp_.device)
                loss_e_proto = torch.zeros((), device=lp_.device)
        else:
            loss_kd = kl_loss(log_prob_student, log_prob_teacher, umask)
            loss_proto = proto_loss(r_s_v, r_t_v, lab_v, proto_t)
            loss_e_proto = proto_loss(hf_s_v['e'], hf_t_v['e'], lab_v, proto_t_e)

            # exp(log_prob) recovers the softmax exactly (no temperature here,
            # unlike kl_loss), so OT needs no raw logits from the model.
            P = lp_t_[valid].exp()
            Q = lp_[valid].exp()
            loss_ot = ot_loss(P, Q, ot_cost)

        # Structural KD: how 'e' relates to a/t/v, within each model own
        # space. Label-free, so it is the one cluster regression leaves alone.
        loss_node_rel = node_rel_loss(hf_s_v, hf_t_v)

        # 4 clusters: CE (always weight 1), prob = KD+OT, proto, rel =
        # e_proto+node_rel. proto/rel are cosine-based and run far smaller
        # than CE (~270x and ~85x on MELD, ~90x and ~45x on IEMOCAP), so
        # --loss-norm calibrated rescales them to CE's scale first and lets
        # the lambdas stay in [0, 1]. Scales are 1.0 when norm is off.
        loss = (loss_ce + args.lambda_prob * (loss_kd + loss_ot)
                + args.lambda_proto * getattr(args, 'proto_scale', 1.0) * loss_proto
                + args.lambda_rel * getattr(args, 'rel_scale', 1.0) * (loss_e_proto + loss_node_rel))

        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())
        losses_ce.append(loss_ce.item() * masks[-1].sum())
        losses_kd.append(loss_kd.item() * masks[-1].sum())
        losses_proto.append(loss_proto.item() * masks[-1].sum())
        losses_ot.append(loss_ot.item() * masks[-1].sum())
        losses_e_proto.append(loss_e_proto.item() * masks[-1].sum())
        losses_node_rel.append(loss_node_rel.item() * masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_loss_ce = round(np.sum(losses_ce) / np.sum(masks), 4)
    avg_loss_kd = round(np.sum(losses_kd) / np.sum(masks), 4)
    avg_loss_proto = round(np.sum(losses_proto) / np.sum(masks), 4)
    avg_loss_ot = round(np.sum(losses_ot) / np.sum(masks), 4)
    avg_loss_e_proto = round(np.sum(losses_e_proto) / np.sum(masks), 4)
    avg_loss_node_rel = round(np.sum(losses_node_rel) / np.sum(masks), 4)
    if getattr(args, 'task', 'classification') == 'regression':
        avg_acc, avg_f1, mae, corr = regression_metrics(labels, preds.reshape(-1), masks)
        extra = {'mae': mae, 'corr': corr}
    else:
        preds = np.argmax(preds, 1)
        avg_acc = accuracy_score(labels, preds, sample_weight=masks)
        avg_f1 = f1_score(labels, preds, sample_weight=masks, average='weighted')
        extra = {}

    out = (avg_acc, avg_f1, avg_loss, avg_loss_ce, avg_loss_kd, avg_loss_proto, avg_loss_ot,
           avg_loss_e_proto, avg_loss_node_rel)
    return out + (extra,) if return_extra else out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', help='IEMOCAPFour, IEMOCAPSix or MELD')
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')

    # ---- architecture: MUST match the teacher checkpoint (see module docstring) ---- #
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--d-e', type=int, default=None)

    # ---- architecture: free to differ from the teacher (no shape impact) ---- #
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)

    # ---- student-only: missing-modality handling ---- #
    parser.add_argument('--prompt-type', type=str, default='prototype', choices=['none', 'constant', 'prototype'])
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)
    parser.add_argument('--mask-type', type=str, default='constant-0.1',
                        help='Missing-rate for the student, as constant-<rate>.')

    # distillation: CE is always weight 1; prob/proto/rel are tunable
    parser.add_argument('--kd-temperature', type=float, default=2.0)
    parser.add_argument('--ot-epsilon', type=float, default=0.1, help='Sinkhorn entropic regularization.')
    parser.add_argument('--ot-iters', type=int, default=50, help='Sinkhorn iterations.')
    parser.add_argument('--lambda-prob', type=float, default=1.0,
                        help='Weight for the prob cluster (KD + OT). 0 disables it. Defaults to 1.0 (original fixed-weight behavior).')
    parser.add_argument('--lambda-proto', type=float, default=1.0,
                        help='Weight for the proto cluster (r vs class prototypes). 0 disables it.')
    parser.add_argument('--lambda-rel', type=float, default=1.0,
                        help="Weight for the rel cluster (e_proto + node_rel, both about the 'e' node). 0 disables it.")

    # ---- training (free to differ from the teacher) ---- #
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer for why adamw may be worth testing.')

    args = parser.parse_args()
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    mask_rate = float(args.mask_type.split('-')[-1])

    cuda = torch.cuda.is_available() and not args.no_cuda
    set_seed(args.seed, cuda)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    save_log = os.path.join(path.LOG_DIR, 'student', f'{args.dataset}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f"{save_log}/{time_now}_student_{args.dataset}.txt", stream=sys.stdout)

    if args.dataset == 'IEMOCAPFour':
        args.num_folder, args.n_classes, args.n_speakers = 5, 4, 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder, args.n_classes, args.n_speakers = 5, 6, 2
    elif args.dataset == 'MELD':
        args.num_folder, args.n_classes, args.n_speakers = 1, 7, 9
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    print(args)

    print('\n========== Reading Data ==========\n')
    if args.dataset == 'MELD':
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=0, meld_path=path.PATH_MELD
        )
    else:
        audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
        text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
        video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            audio_root=audio_root, text_root=text_root, video_root=video_root,
            num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
        )

    print(f'\n========== Training Student (mask_rate={mask_rate}, prompt_type={args.prompt_type}, '
          f'lambda_prob={args.lambda_prob}, lambda_proto={args.lambda_proto}, lambda_rel={args.lambda_rel} '
          f'(CE fixed=1), per fold) ==========\n')

    fold_f1, fold_acc = [], []

    for ii in range(args.num_folder):
        print(f'>>>>> Student fold {ii+1}/{args.num_folder} >>>>>')
        start_time = time.time()

        # The teacher was built with prompt_type='none', so it must be
        # rebuilt that way here too regardless of the student's setting --
        # otherwise load_state_dict hits a prompt-key mismatch.
        teacher_args = copy.deepcopy(args)
        teacher_args.prompt_type = 'none'
        teacher = build_model(teacher_args, adim, tdim, vdim)
        teacher_state = load_teacher_checkpoint(args.dataset, fold_index=ii, map_location=args.device)
        teacher.load_state_dict(teacher_state)
        if cuda:
            teacher.cuda()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        # Prototypes and the OT cost derived from them are fixed per fold:
        # computed once here, reused by every batch.
        proto_t, proto_t_e = compute_teacher_prototypes(args, teacher, train_loaders[ii], adim, tdim, vdim)
        ot_cost = compute_ot_cost_matrix(proto_t)
        print(f'  teacher prototypes computed: shape={tuple(proto_t.shape)}, '
              f'proto_t_e shape={tuple(proto_t_e.shape)}, ot_cost shape={tuple(ot_cost.shape)}')

        student = build_model(args, adim, tdim, vdim)
        if cuda:
            student.cuda()

        cls_loss = MaskedCELoss()
        kl_loss = MaskedKLLoss(temperature=args.kd_temperature)
        proto_loss = PrototypeRelationLoss()
        ot_loss = SinkhornOTLoss(epsilon=args.ot_epsilon, n_iters=args.ot_iters)
        node_rel_loss = NodeRelationLoss()
        if cuda:
            cls_loss.cuda()

        optimizer = make_optimizer(student.parameters(), args)

        best_val_fscore = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            (train_acc, train_fscore, train_loss, train_ce, train_kd, train_proto, train_ot,
             train_e_proto, train_node_rel) = train_or_eval_student(
                args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                ot_loss, ot_cost, node_rel_loss,
                train_loaders[ii], mask_rate=mask_rate, optimizer=optimizer, train=True)
            (val_acc, val_fscore, val_loss, val_ce, val_kd, val_proto, val_ot,
             val_e_proto, val_node_rel) = train_or_eval_student(
                args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                ot_loss, ot_cost, node_rel_loss,
                val_loaders[ii], mask_rate=mask_rate, optimizer=None, train=False)

            print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; '
                  f'train_loss:{train_loss} (ce={train_ce}, kd={train_kd}, proto={train_proto}, ot={train_ot}, '
                  f'e_proto={train_e_proto}, node_rel={train_node_rel}); '
                  f'val_fscore:{val_fscore:2.2%}; val_acc:{val_acc:2.2%}; '
                  f'val_loss:{val_loss} (ce={val_ce}, kd={val_kd}, proto={val_proto}, ot={val_ot}, '
                  f'e_proto={val_e_proto}, node_rel={val_node_rel})')

            if val_fscore > best_val_fscore + args.early_stop_delta:
                best_val_fscore = val_fscore
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(student.state_dict())
                print(f"  New best student val F1: {best_val_fscore:.4f}.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print(f"  Early stopping student after {epoch + 1} epochs.")
                    break

        student.load_state_dict(best_model_state)
        (test_acc, test_fscore, test_loss, test_ce, test_kd, test_proto, test_ot,
         test_e_proto, test_node_rel) = train_or_eval_student(
            args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
            ot_loss, ot_cost, node_rel_loss,
            test_loaders[ii], mask_rate=mask_rate, optimizer=None, train=False)

        print(f'>>>>> Student fold {ii+1}: best val_f1={best_val_fscore:.4f}, '
              f'test_f1={test_fscore:.4f}, test_acc={test_acc:.4f}, '
              f'duration={time.time()-start_time:.2f}s >>>>>')

        fold_f1.append(test_fscore)
        fold_acc.append(test_acc)

        save_student_checkpoint(best_model_state, args.dataset, fold_index=ii)

    print(f'\n====== Student summary: mean test F1={np.mean(fold_f1):.4f}, '
          f'mean test acc={np.mean(fold_acc):.4f} ======')
