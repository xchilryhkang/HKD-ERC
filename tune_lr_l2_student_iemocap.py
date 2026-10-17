"""
Tunes lr x l2 for the STUDENT on IEMOCAPSix, same standard leave-one-
session-out protocol as tune_lr_l2_iemocap.py / train_student_iemocap.py:
all 4 train sessions used, fixed epochs, no validation split. Repeated
across multiple seeds to report mean/std of the 5-fold CV test F1 per
(lr, l2) combo.

The teacher (and its per-fold prototypes / OT cost matrix) is loaded and
computed ONCE PER FOLD, then reused across every (lr, l2, seed) combo for
that fold -- it never changes within this sweep, so recomputing it per
combo would just waste time. Distillation lambdas (kd/proto/ot/e_proto/
node_rel) are all held at their default 1.0 -- not swept here.

Requires teacher checkpoints for the dataset to already exist under
output/model/teacher/ (run train_teacher_iemocap.py / run_teacher_iemocap6.*
first).

Isolated and deletable: only imports from train_student.py / utils.py /
kd_losses.py / loss.py, and nothing imports this file back. Results are
written to their own .txt file under output/tune/, plus printed to the
console.
"""

import os
import sys
import time
import argparse
import datetime
import copy

import numpy as np
import torch
import torch.optim as optim

import path
from utils import get_loaders, build_model, load_teacher_checkpoint
from loss import MaskedCELoss
from kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)
from train_student import compute_teacher_prototypes, train_or_eval_student


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lrs', type=str, default='0.00005,0.0001,0.0002,0.0005',
                        help='Comma-separated learning rate candidates.')
    parser.add_argument('--l2s', type=str, default='0,0.00001,0.0001',
                        help='Comma-separated weight-decay (L2) candidates.')
    parser.add_argument('--seeds', type=str, default='197,649,722',
                        help='Comma-separated training seeds.')
    parser.add_argument('--epochs', type=int, default=40, help='Fixed epoch count per fold (no early stopping).')

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', choices=['IEMOCAPFour', 'IEMOCAPSix'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)

    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--d-e', type=int, default=None)

    parser.add_argument('--prompt-type', type=str, default='none', choices=['none', 'constant', 'prototype'])
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)
    parser.add_argument('--mask-type', type=str, default='constant-0.1')

    parser.add_argument('--kd-temperature', type=float, default=2.0)
    parser.add_argument('--ot-epsilon', type=float, default=0.1)
    parser.add_argument('--ot-iters', type=int, default=50)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    lrs = [float(x) for x in args.lrs.split(',')]
    l2s = [float(x) for x in args.l2s.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.lambda_proto = 1.0
    args.lambda_rel = 1.0
    mask_rate = float(args.mask_type.split('-')[-1])
    args.num_folder, args.n_classes, args.n_speakers = (
        (5, 4, 2) if args.dataset == 'IEMOCAPFour' else (5, 6, 2)
    )
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available() and not args.no_cuda

    n_combos = len(lrs) * len(l2s)

    out_dir = os.path.join(path.SAVED_ROOT, 'tune')
    os.makedirs(out_dir, exist_ok=True)
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    result_path = os.path.join(out_dir, f'{time_now}_lr_l2_sweep_student_iemocap_noval_{args.dataset}.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'student lr x l2 tune (standard LOSO-CV, no val, fixed epochs) — dataset={args.dataset}')
    log(f'lrs={lrs}  l2s={l2s}  ({n_combos} combos)  training_seeds={seeds}  epochs={args.epochs}  '
        f'mask_rate={mask_rate}  prompt_type={args.prompt_type}')
    log(f'lambdas: CE/prob fixed=1, proto=rel=1.0 (fixed, not swept here)')
    log('-' * 100)

    print('\n========== Reading Data ==========\n')
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), 'features not exist!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
    )

    print('\n========== Loading teacher + computing prototypes per fold (once, reused across the whole sweep) ==========\n')
    teachers, proto_ts, proto_t_es, ot_costs = [], [], [], []
    teacher_args = copy.deepcopy(args)
    teacher_args.prompt_type = 'none'
    for ii in range(args.num_folder):
        teacher = build_model(teacher_args, adim, tdim, vdim)
        teacher_state = load_teacher_checkpoint(args.dataset, fold_index=ii, map_location=args.device)
        teacher.load_state_dict(teacher_state)
        if cuda:
            teacher.cuda()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        proto_t, proto_t_e = compute_teacher_prototypes(args, teacher, train_loaders[ii], adim, tdim, vdim)
        ot_cost = compute_ot_cost_matrix(proto_t)

        teachers.append(teacher)
        proto_ts.append(proto_t)
        proto_t_es.append(proto_t_e)
        ot_costs.append(ot_cost)
        print(f'  fold {ii}: teacher loaded, proto_t shape={tuple(proto_t.shape)}')

    results = []
    combo_idx = 0

    for lr in lrs:
        for l2 in l2s:
            combo_idx += 1
            log(f'\n[{combo_idx}/{n_combos}] lr={lr} l2={l2}')

            seed_test_means = []
            combo_start = time.time()

            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                if cuda:
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                fold_test_f1 = []
                for ii in range(args.num_folder):
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

                    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=l2)

                    for epoch in range(args.epochs):
                        train_or_eval_student(
                            args, teachers[ii], student, cls_loss, kl_loss, proto_loss,
                            proto_ts[ii], proto_t_es[ii], ot_loss, ot_costs[ii], node_rel_loss,
                            train_loaders[ii], mask_rate=mask_rate, optimizer=optimizer, train=True)

                    (test_acc, test_fscore, test_loss, test_ce, test_kd, test_proto, test_ot,
                     test_e_proto, test_node_rel) = train_or_eval_student(
                        args, teachers[ii], student, cls_loss, kl_loss, proto_loss,
                        proto_ts[ii], proto_t_es[ii], ot_loss, ot_costs[ii], node_rel_loss,
                        test_loaders[ii], mask_rate=mask_rate, optimizer=None, train=False)

                    log(f'    seed={seed} fold={ii} test_f1={test_fscore:.4f}')
                    fold_test_f1.append(test_fscore)

                seed_test_mean = float(np.mean(fold_test_f1))
                seed_test_means.append(seed_test_mean)
                log(f'  -> seed={seed}: 5-fold mean test_f1={seed_test_mean:.4f}')

            mean_test = float(np.mean(seed_test_means))
            std_test = float(np.std(seed_test_means))
            duration = time.time() - combo_start

            log(f'>>>>> lr={lr} l2={l2}: test_f1 mean={mean_test:.4f} std={std_test:.4f} | '
                f'duration={duration:.1f}s >>>>>')

            results.append((lr, l2, mean_test, std_test, duration))

    log('\n===== SUMMARY (sorted by mean test_f1, best first) =====')
    for lr, l2, mean_test, std_test, duration in sorted(results, key=lambda r: -r[2]):
        log(f'lr={lr:<10} l2={l2:<10}  test_f1={mean_test:.4f}±{std_test:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
