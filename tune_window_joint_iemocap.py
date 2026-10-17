"""
Tunes windowp/windowf (symmetric, windowp=windowf=w) JOINTLY for teacher +
student on IEMOCAPSix: for each window value, a teacher is trained FRESH
with that window (5-fold leave-one-session-out CV, best checkpoint = best
session-out F1, early stopping tracks the held-out session directly every
epoch -- see train_teacher_iemocap.py's module docstring), then that
teacher is IMMEDIATELY used (in-memory, nothing saved to disk) to distill
a student with that SAME window, same early-stopping-on-session-out
protocol. The window is selected by the STUDENT's best session-out F1,
not the teacher's own -- a window that's best for the teacher alone isn't
necessarily the window that distills best into the student.

lr/l2 are held FIXED at the values tune_lr_l2_iemocap.py found best
(lr=0.0001, l2=0.0001) for BOTH teacher and student. Repeated across
multiple seeds to report mean/std of the 5-fold CV session-out F1 per
window value.

Per (window, seed, fold): train teacher to its own best session-out epoch
-> train student (distilling from that teacher) to ITS best session-out
epoch -> record the student's best session-out F1. Both teacher and
student are retrained from scratch every single time (nothing is cached
across window values, since window changes the graph structure itself).

No checkpoints are saved anywhere in this sweep -- once the best window is
found, train_teacher_iemocap.py then train_student_iemocap.py need to be
re-run once each with that window to produce the real, saved checkpoints.

Isolated and deletable: only imports from train_teacher.py / train_student.py
/ utils.py / kd_losses.py / loss.py, and nothing imports this file back.
Results are written to their own .txt file under output/tune/, plus
printed to the console.
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
from utils import get_loaders, build_model
from loss import MaskedCELoss
from kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)
from train_teacher import train_or_eval_model
from train_student import compute_teacher_prototypes, train_or_eval_student


def train_teacher_to_best(teacher_args, adim, tdim, vdim, train_loader, sessionout_loader, cuda):
    teacher = build_model(teacher_args, adim, tdim, vdim)
    cls_loss = MaskedCELoss()
    if cuda:
        teacher.cuda()
        cls_loss.cuda()
    optimizer = optim.Adam(teacher.parameters(), lr=teacher_args.lr, weight_decay=teacher_args.l2)

    best_f1, best_acc, best_state, no_improve = 0.0, 0.0, None, 0
    for epoch in range(teacher_args.epochs):
        train_or_eval_model(teacher_args, teacher, cls_loss, train_loader, mask_rate=0.0,
                             optimizer=optimizer, train=True)
        so_acc, so_f1, _ = train_or_eval_model(
            teacher_args, teacher, cls_loss, sessionout_loader, mask_rate=0.0, optimizer=None, train=False)
        if so_f1 > best_f1 + teacher_args.early_stop_delta:
            best_f1, best_acc, no_improve = so_f1, so_acc, 0
            best_state = copy.deepcopy(teacher.state_dict())
        else:
            no_improve += 1
            if no_improve >= teacher_args.early_stop_patience:
                break
    teacher.load_state_dict(best_state)
    return teacher, best_f1, best_acc


def train_student_to_best(args, teacher, adim, tdim, vdim, train_loader, sessionout_loader, mask_rate, cuda):
    proto_t, proto_t_e = compute_teacher_prototypes(args, teacher, train_loader, adim, tdim, vdim)
    ot_cost = compute_ot_cost_matrix(proto_t)

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
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.l2)

    best_f1, best_acc, no_improve = 0.0, 0.0, 0
    for epoch in range(args.epochs):
        train_or_eval_student(
            args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
            ot_loss, ot_cost, node_rel_loss, train_loader, mask_rate=mask_rate, optimizer=optimizer, train=True)
        result = train_or_eval_student(
            args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
            ot_loss, ot_cost, node_rel_loss, sessionout_loader, mask_rate=mask_rate, optimizer=None, train=False)
        so_acc, so_f1 = result[0], result[1]
        if so_f1 > best_f1 + args.early_stop_delta:
            best_f1, best_acc, no_improve = so_f1, so_acc, 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                break
    return best_f1, best_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--windows', type=str, default='1,2,3,4,5',
                        help='Comma-separated window candidates (windowp=windowf=w for each, shared by teacher+student).')
    parser.add_argument('--lr', type=float, default=0.0001, help='Fixed lr (from tune_lr_l2_iemocap.py), used for both.')
    parser.add_argument('--l2', type=float, default=0.0001, help='Fixed l2 (from tune_lr_l2_iemocap.py), used for both.')
    parser.add_argument('--seeds', type=str, default='197,649,722',
                        help='Comma-separated training seeds.')
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks session-out F1.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--mask-rate', type=float, default=0.2, help='Student missing-modality rate.')
    parser.add_argument('--prompt-type', type=str, default='none', choices=['none', 'constant', 'prototype'])

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', choices=['IEMOCAPFour', 'IEMOCAPSix'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')

    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--d-e', type=int, default=None)
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)

    parser.add_argument('--kd-temperature', type=float, default=2.0)
    parser.add_argument('--ot-epsilon', type=float, default=0.1)
    parser.add_argument('--ot-iters', type=int, default=50)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    windows = [int(x) for x in args.windows.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.lambda_proto = 1.0
    args.lambda_rel = 1.0
    mask_rate = args.mask_rate
    args.num_folder, args.n_classes, args.n_speakers = (
        (5, 4, 2) if args.dataset == 'IEMOCAPFour' else (5, 6, 2)
    )
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available() and not args.no_cuda

    n_combos = len(windows)

    out_dir = os.path.join(path.SAVED_ROOT, 'tune')
    os.makedirs(out_dir, exist_ok=True)
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    result_path = os.path.join(out_dir, f'{time_now}_window_sweep_joint_iemocap_{args.dataset}.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'window JOINT tune (teacher -> immediately distill student, both by best session-out F1, '
        f'selected by STUDENT) — dataset={args.dataset}')
    log(f'windows={windows} (windowp=windowf=w, shared)  ({n_combos} combos)  training_seeds={seeds}  '
        f'epochs_cap={args.epochs}  early_stop_patience={args.early_stop_patience}  '
        f'mask_rate={mask_rate}  prompt_type={args.prompt_type}')
    log(f'lr={args.lr} (fixed, both)  l2={args.l2} (fixed, both)  lambdas: CE/prob fixed=1, proto=rel=1.0 (fixed)')
    log('-' * 100)

    print('\n========== Reading Data ==========\n')
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), 'features not exist!'
    train_loaders, _unused_val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
    )

    results = []
    combo_idx = 0

    for w in windows:
        combo_idx += 1
        args.windowp = w
        args.windowf = w
        log(f'\n[{combo_idx}/{n_combos}] windowp=windowf={w}')

        seed_teacher_means, seed_student_means = [], []
        combo_start = time.time()

        for seed in seeds:
            fold_teacher_f1, fold_student_f1 = [], []

            for ii in range(args.num_folder):
                np.random.seed(seed)
                torch.manual_seed(seed)
                if cuda:
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                teacher_args = copy.deepcopy(args)
                teacher_args.prompt_type = 'none'
                teacher, teacher_f1, teacher_acc = train_teacher_to_best(
                    teacher_args, adim, tdim, vdim, train_loaders[ii], test_loaders[ii], cuda)
                fold_teacher_f1.append(teacher_f1)

                for p in teacher.parameters():
                    p.requires_grad_(False)
                teacher.eval()

                student_f1, student_acc = train_student_to_best(
                    args, teacher, adim, tdim, vdim, train_loaders[ii], test_loaders[ii], mask_rate, cuda)
                fold_student_f1.append(student_f1)

                log(f'    seed={seed} fold={ii} teacher_f1={teacher_f1:.4f} student_f1={student_f1:.4f}')

            seed_teacher_mean = float(np.mean(fold_teacher_f1))
            seed_student_mean = float(np.mean(fold_student_f1))
            seed_teacher_means.append(seed_teacher_mean)
            seed_student_means.append(seed_student_mean)
            log(f'  -> seed={seed}: 5-fold mean teacher_f1={seed_teacher_mean:.4f} '
                f'student_f1={seed_student_mean:.4f}')

        mean_teacher = float(np.mean(seed_teacher_means))
        std_teacher = float(np.std(seed_teacher_means))
        mean_student = float(np.mean(seed_student_means))
        std_student = float(np.std(seed_student_means))
        duration = time.time() - combo_start

        log(f'>>>>> window={w}: teacher_f1 mean={mean_teacher:.4f} std={std_teacher:.4f} | '
            f'student_f1 mean={mean_student:.4f} std={std_student:.4f} | duration={duration:.1f}s >>>>>')

        results.append((w, mean_teacher, std_teacher, mean_student, std_student, duration))

    log('\n===== SUMMARY (sorted by STUDENT mean session-out F1, best first) =====')
    for w, mean_teacher, std_teacher, mean_student, std_student, duration in sorted(results, key=lambda r: -r[3]):
        log(f'window={w:<4}  student_f1={mean_student:.4f}±{std_student:.4f}  '
            f'teacher_f1={mean_teacher:.4f}±{std_teacher:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
