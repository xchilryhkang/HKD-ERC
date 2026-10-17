"""
Sweeps lambda_rel x lambda_prob for the student on IEMOCAPSix, 5-fold
leave-one-session-out CV, best checkpoint = best session-out F1 (same
early-stopping convention as train_student_iemocap.py). lambda_proto is
left at its default (1.0, not swept). Distills from the already-tuned
teacher checkpoints (output/model/teacher/, lr=0.0002/l2=1e-5/adamw) via
utils.load_teacher_checkpoint -- run train_teacher_iemocap.py first if
those don't exist yet.

Grid values default to 0.2/0.4/0.6/0.8/1.0 for both lambdas (5x5 = 25
combos). For plotting as a 3D bar chart (WAF/F1 vs lambda1 x lambda2,
style matching an existing CMU-MOSI/IEMOCAP reference figure):
lambda1 = lambda_rel, lambda2 = lambda_prob (user-confirmed axis mapping).

Supports --max-combos (stop after N combos, e.g. to run a partial sweep
now and continue later) and --skip-combos/--append-to (resume mid-sweep,
same convention as tune_lr_l2_iemocap.py).

Isolated and deletable: only imports from train_student.py / utils.py /
loss.py / kd_losses.py, and nothing imports this file back. Results are
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

# Make the repo root importable so `erc` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erc import path
from erc.utils import get_loaders, build_model, load_teacher_checkpoint, make_optimizer, set_seed
from erc.losses.loss import MaskedCELoss
from erc.losses.kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)
from train.train_student import compute_teacher_prototypes, train_or_eval_student


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda-rels', type=str, default='0.2,0.4,0.6,0.8,1.0',
                        help='Comma-separated lambda_rel candidates (lambda1 in the 3D plot).')
    parser.add_argument('--lambda-probs', type=str, default='0.2,0.4,0.6,0.8,1.0',
                        help='Comma-separated lambda_prob candidates (lambda2 in the 3D plot).')
    parser.add_argument('--lambda-proto', type=float, default=1.0, help='Held fixed, not swept.')
    parser.add_argument('--seeds', type=str, default='50', help='Comma-separated training seeds.')
    parser.add_argument('--mask-rate', type=float, default=0.3, help="Student's missing rate at train+eval time.")
    parser.add_argument('--prompt-type', type=str, default='none', choices=['none', 'constant', 'prototype'])
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--l2', type=float, default=0.00001)

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', choices=['IEMOCAPFour', 'IEMOCAPSix'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks session-out F1.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--d-e', type=int, default=None)
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)
    parser.add_argument('--kd-temperature', type=float, default=2.0)
    parser.add_argument('--ot-epsilon', type=float, default=0.1)
    parser.add_argument('--ot-iters', type=int, default=50)

    parser.add_argument('--max-combos', type=int, default=None,
                        help='Stop after this many combos (resume later with --skip-combos).')
    parser.add_argument('--skip-combos', type=int, default=0,
                        help='Skip the first N (lambda1,lambda2) combos in iteration order (resume mid-sweep).')
    parser.add_argument('--append-to', type=str, default=None,
                        help='Existing log file to append to, instead of creating a new one (resume mid-sweep).')

    args = parser.parse_args()
    lambda_rels = [float(x) for x in args.lambda_rels.split(',')]
    lambda_probs = [float(x) for x in args.lambda_probs.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    args.mask_type = f'constant-{args.mask_rate}'
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.num_folder, args.n_classes, args.n_speakers = (
        (5, 4, 2) if args.dataset == 'IEMOCAPFour' else (5, 6, 2)
    )
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available() and not args.no_cuda

    n_combos = len(lambda_rels) * len(lambda_probs)

    if args.append_to:
        result_path = args.append_to
        assert os.path.exists(result_path), f'--append-to path not found: {result_path}'
    else:
        out_dir = os.path.join(path.SAVED_ROOT, 'tune')
        os.makedirs(out_dir, exist_ok=True)
        time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        result_path = os.path.join(out_dir, f'{time_now}_lambda_prob_rel_sweep_iemocap_{args.dataset}.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    if args.skip_combos:
        log(f'\n===== RESUMING from combo {args.skip_combos + 1}/{n_combos} (skipping first {args.skip_combos}) =====')
    else:
        log(f'lambda_rel (lambda1) x lambda_prob (lambda2) tune (student, IEMOCAP 5-fold '
            f'leave-one-session-out CV, best=session-out F1) — dataset={args.dataset}  '
            f'mask_rate={args.mask_rate}  prompt_type={args.prompt_type}')
        log(f'lambda_rels={lambda_rels}  lambda_probs={lambda_probs}  ({n_combos} combos)  '
            f'lambda_proto(fixed)={args.lambda_proto}  training_seeds={seeds}  '
            f'lr={args.lr}  l2={args.l2}  optimizer={args.optimizer}  '
            f'epochs_cap={args.epochs}  early_stop_patience={args.early_stop_patience}')
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

    # Teacher is frozen and identical across every combo/seed -- load once per
    # fold, outside the sweep loops, instead of reloading it per combo.
    print('\n========== Loading teacher checkpoints + computing prototypes (once per fold) ==========\n')
    teacher_args = copy.deepcopy(args)
    teacher_args.prompt_type = 'none'
    teachers, protos_t, protos_t_e, ot_costs = [], [], [], []
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
        protos_t.append(proto_t)
        protos_t_e.append(proto_t_e)
        ot_costs.append(ot_cost)
        print(f'  fold {ii}: proto_t={tuple(proto_t.shape)}, proto_t_e={tuple(proto_t_e.shape)}')

    results = []
    combo_idx = 0

    for l1 in lambda_rels:          # lambda1 = lambda_rel
        for l2 in lambda_probs:     # lambda2 = lambda_prob
            combo_idx += 1
            if combo_idx <= args.skip_combos:
                continue
            if args.max_combos is not None and (combo_idx - args.skip_combos) > args.max_combos:
                log(f'\n===== STOPPING after {args.max_combos} combos this run '
                    f'(resume with --skip-combos {combo_idx - 1}) =====')
                break
            log(f'\n[{combo_idx}/{n_combos}] lambda_rel(l1)={l1} lambda_prob(l2)={l2}')

            args.lambda_rel = l1
            args.lambda_prob = l2
            seed_means = []
            combo_start = time.time()

            for seed in seeds:
                set_seed(seed, cuda)

                fold_f1 = []
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
                    optimizer = make_optimizer(student.parameters(), args)

                    best_f1, no_improve = 0.0, 0
                    for epoch in range(args.epochs):
                        train_or_eval_student(
                            args, teachers[ii], student, cls_loss, kl_loss, proto_loss,
                            protos_t[ii], protos_t_e[ii], ot_loss, ot_costs[ii], node_rel_loss,
                            train_loaders[ii], mask_rate=args.mask_rate, optimizer=optimizer, train=True)
                        (so_acc, so_f1, so_loss, so_ce, so_kd, so_proto, so_ot,
                         so_e_proto, so_node_rel) = train_or_eval_student(
                            args, teachers[ii], student, cls_loss, kl_loss, proto_loss,
                            protos_t[ii], protos_t_e[ii], ot_loss, ot_costs[ii], node_rel_loss,
                            test_loaders[ii], mask_rate=args.mask_rate, optimizer=None, train=False)
                        if so_f1 > best_f1 + args.early_stop_delta:
                            best_f1, no_improve = so_f1, 0
                        else:
                            no_improve += 1
                            if no_improve >= args.early_stop_patience:
                                break

                    log(f'    seed={seed} fold={ii} sessionout_f1={best_f1:.4f}')
                    fold_f1.append(best_f1)

                seed_mean = float(np.mean(fold_f1))
                seed_means.append(seed_mean)
                log(f'  -> seed={seed}: 5-fold mean sessionout_f1={seed_mean:.4f}')

            mean_f1, std_f1 = float(np.mean(seed_means)), float(np.std(seed_means))
            duration = time.time() - combo_start

            log(f'>>>>> lambda_rel={l1} lambda_prob={l2}: sessionout_f1 mean={mean_f1:.4f} std={std_f1:.4f} | '
                f'duration={duration:.1f}s >>>>>')

            results.append((l1, l2, mean_f1, std_f1, duration))
        else:
            continue
        break

    log('\n===== SUMMARY (sorted by mean sessionout_f1, best first) =====')
    for l1, l2, mean_f1, std_f1, duration in sorted(results, key=lambda r: -r[2]):
        log(f'lambda_rel={l1:<5} lambda_prob={l2:<5}  sessionout_f1={mean_f1:.4f}±{std_f1:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
