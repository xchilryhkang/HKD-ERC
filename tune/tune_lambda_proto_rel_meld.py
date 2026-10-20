import os
import time
import argparse
import datetime
import copy

import numpy as np
import torch

# Make the repo root importable so `src` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import path
from src.utils import get_loaders_meld, build_model, load_teacher_checkpoint, make_optimizer, set_seed
from src.losses.loss import MaskedCELoss
from src.losses.kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)
from train.train_student import compute_teacher_prototypes, train_or_eval_student


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda-protos', type=str, default='1,10,50,100,200',
                        help='lambda_proto candidates (lambda1 in the 3D plot). proto runs ~0.004x CE, '
                             'so these map to roughly 0.2%%/2%%/10%%/19%%/32%% of total loss.')
    parser.add_argument('--lambda-rels', type=str, default='1,5,25,50,100',
                        help='lambda_rel candidates (lambda2 in the 3D plot). rel runs ~0.012x CE, '
                             'so these map to roughly 0.7%%/4%%/16%%/27%%/43%% of total loss.')
    parser.add_argument('--lambda-prob', type=float, default=1.0,
                        help='Held fixed at 1: prob is already CE-scale (~0.63x), so it needs no rebalancing.')
    parser.add_argument('--seeds', type=str, default='50', help='Comma-separated training seeds.')
    parser.add_argument('--mask-rate', type=float, default=0.1, help="Student's missing rate at train+eval time.")
    parser.add_argument('--prompt-type', type=str, default='none', choices=['none', 'constant', 'prototype'])
    parser.add_argument('--roberta-idx', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--l2', type=float, default=0.0001)

    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--d-e', type=int, default=None)
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)
    parser.add_argument('--kd-temperature', type=float, default=2.0)
    parser.add_argument('--ot-epsilon', type=float, default=0.1)
    parser.add_argument('--ot-iters', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks the real valid F1.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--max-combos', type=int, default=None,
                        help='Stop after this many combos (resume later with --skip-combos).')
    parser.add_argument('--skip-combos', type=int, default=0,
                        help='Skip the first N (lambda1,lambda2) combos in iteration order (resume mid-sweep).')
    parser.add_argument('--append-to', type=str, default=None,
                        help='Existing log file to append to, instead of creating a new one (resume mid-sweep).')

    args = parser.parse_args()
    lambda_protos = [float(x) for x in args.lambda_protos.split(',')]
    lambda_rels = [float(x) for x in args.lambda_rels.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    args.dataset = 'MELD'
    args.num_folder = 1
    args.n_classes = 7
    args.n_speakers = 9
    args.mask_type = f'constant-{args.mask_rate}'
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available() and not args.no_cuda

    n_combos = len(lambda_protos) * len(lambda_rels)

    if args.append_to:
        result_path = args.append_to
        assert os.path.exists(result_path), f'--append-to path not found: {result_path}'
    else:
        out_dir = os.path.join(path.SAVED_ROOT, 'tune')
        os.makedirs(out_dir, exist_ok=True)
        time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        result_path = os.path.join(out_dir, f'{time_now}_lambda_proto_rel_sweep_meld.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    if args.skip_combos:
        log(f'\n===== RESUMING from combo {args.skip_combos + 1}/{n_combos} (skipping first {args.skip_combos}) =====')
    else:
        log(f'lambda_proto (lambda1) x lambda_rel (lambda2) tune (student, MELD genuine '
            f'train/valid/test split, best=valid F1, test F1 reported at best-val checkpoint) — '
            f'mask_rate={args.mask_rate}  prompt_type={args.prompt_type}')
        log(f'lambda_protos={lambda_protos}  lambda_rels={lambda_rels}  ({n_combos} combos)  '
            f'lambda_prob(fixed)={args.lambda_prob}  training_seeds={seeds}  '
            f'lr={args.lr}  l2={args.l2}  optimizer={args.optimizer}  roberta_idx={args.roberta_idx}  '
            f'epochs_cap={args.epochs}  early_stop_patience={args.early_stop_patience}')
        log('-' * 100)

    print('\n========== Reading Data ==========\n')
    assert os.path.exists(path.PATH_MELD), 'PATH_MELD not found!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders_meld(
        features_path=path.PATH_MELD,
        batch_size=args.batch_size, num_workers=0, roberta_idx=args.roberta_idx
    )
    train_loader, val_loader, test_loader = train_loaders[0], val_loaders[0], test_loaders[0]

    print('\n========== Loading teacher checkpoint + computing prototypes (once) ==========\n')
    teacher_args = copy.deepcopy(args)
    teacher_args.prompt_type = 'none'
    teacher = build_model(teacher_args, adim, tdim, vdim)
    teacher_state = load_teacher_checkpoint(args.dataset, fold_index=0, map_location=args.device)
    teacher.load_state_dict(teacher_state)
    if cuda:
        teacher.cuda()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    proto_t, proto_t_e = compute_teacher_prototypes(args, teacher, train_loader, adim, tdim, vdim)
    ot_cost = compute_ot_cost_matrix(proto_t)
    print(f'  proto_t={tuple(proto_t.shape)}, proto_t_e={tuple(proto_t_e.shape)}, ot_cost={tuple(ot_cost.shape)}')

    results = []
    combo_idx = 0

    for l1 in lambda_protos:        # lambda1 = lambda_proto
        for l2 in lambda_rels:      # lambda2 = lambda_rel
            combo_idx += 1
            if combo_idx <= args.skip_combos:
                continue
            if args.max_combos is not None and (combo_idx - args.skip_combos) > args.max_combos:
                log(f'\n===== STOPPING after {args.max_combos} combos this run '
                    f'(resume with --skip-combos {combo_idx - 1}) =====')
                break
            log(f'\n[{combo_idx}/{n_combos}] lambda_proto(l1)={l1} lambda_rel(l2)={l2}')

            args.lambda_proto = l1
            args.lambda_rel = l2
            seed_val, seed_test = [], []
            combo_start = time.time()

            for seed in seeds:
                set_seed(seed, cuda)

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

                best_val_f1, best_state, no_improve = 0.0, None, 0
                for epoch in range(args.epochs):
                    train_or_eval_student(
                        args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                        ot_loss, ot_cost, node_rel_loss,
                        train_loader, mask_rate=args.mask_rate, optimizer=optimizer, train=True)
                    (val_acc, val_f1, val_loss, val_ce, val_kd, val_proto, val_ot,
                     val_e_proto, val_node_rel) = train_or_eval_student(
                        args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                        ot_loss, ot_cost, node_rel_loss,
                        val_loader, mask_rate=args.mask_rate, optimizer=None, train=False)
                    if val_f1 > best_val_f1 + args.early_stop_delta:
                        best_val_f1, no_improve = val_f1, 0
                        best_state = copy.deepcopy(student.state_dict())
                    else:
                        no_improve += 1
                        if no_improve >= args.early_stop_patience:
                            break

                student.load_state_dict(best_state)
                (test_acc, test_f1, *_rest) = train_or_eval_student(
                    args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                    ot_loss, ot_cost, node_rel_loss,
                    test_loader, mask_rate=args.mask_rate, optimizer=None, train=False)

                log(f'    seed={seed} val_f1={best_val_f1:.4f} test_f1={test_f1:.4f}')
                seed_val.append(best_val_f1)
                seed_test.append(test_f1)

            mean_val, std_val = float(np.mean(seed_val)), float(np.std(seed_val))
            mean_test, std_test = float(np.mean(seed_test)), float(np.std(seed_test))
            duration = time.time() - combo_start

            log(f'>>>>> lambda_proto={l1} lambda_rel={l2}: val_f1 mean={mean_val:.4f} std={std_val:.4f} | '
                f'test_f1 mean={mean_test:.4f} std={std_test:.4f} | duration={duration:.1f}s >>>>>')

            results.append((l1, l2, mean_val, std_val, mean_test, std_test, duration))
        else:
            continue
        break

    log('\n===== SUMMARY (sorted by mean val_f1, best first) =====')
    for l1, l2, mean_val, std_val, mean_test, std_test, duration in sorted(results, key=lambda r: -r[2]):
        log(f'lambda_proto={l1:<5} lambda_rel={l2:<5}  val_f1={mean_val:.4f}±{std_val:.4f}  '
            f'test_f1={mean_test:.4f}±{std_test:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
