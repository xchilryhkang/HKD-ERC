"""
Trains the student for MELD using a GENUINE validation split (see
train_teacher_meld.py's module docstring for the full data-source
rationale -- same utils.get_loaders_meld, same two merged pkl files).

Early stopping tracks the REAL valid split every epoch and keeps the
checkpoint from whichever epoch scored best on it. All five distillation
losses (KD / proto / OT / e_proto / node_rel) and their lambda_* weights,
kd_temperature, ot_epsilon/ot_iters, prompt_type etc. all work exactly as
in train_student.py / train_student_iemocap.py -- only the data source
(single genuine train/valid/test split, not 5-fold CV) differs.

Isolated and deletable: reuses concat_repr/compute_teacher_prototypes/
train_or_eval_student/etc. from train_student.py and utils.py (which
already branches on args.dataset == 'MELD' for the one-hot-speaker ->
qmask_9 conversion), but nothing imports this file back, so it can be
deleted without side effects. Saves checkpoints via the SAME
save_student_checkpoint(dataset='MELD', fold_index=0) path train_student.py
uses -- running this will overwrite MELD's existing student checkpoint.
Requires a teacher checkpoint for MELD to already exist under
output/model/teacher/ -- run train_teacher_meld.py / run_teacher_meld.ps1
first if it doesn't.
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

# Make the repo root importable so `erc` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erc import path
from erc.utils import (
    Logger, get_loaders_meld, build_model, load_teacher_checkpoint, save_student_checkpoint, make_optimizer, set_seed
)
from erc.losses.loss import MaskedCELoss
from erc.losses.kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)
from train.train_student import compute_teacher_prototypes, train_or_eval_student


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--roberta-idx', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Which of the 4 RoBERTa text variants to use -- MUST match what '
                             'train_teacher_meld.py used for the checkpoint being loaded.')

    # ---- architecture: MUST match the teacher checkpoint (see train_student.py's docstring) ---- #
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
    parser.add_argument('--prompt-type', type=str, default='none', choices=['none', 'constant', 'prototype'])
    parser.add_argument('--n-proto', type=int, default=10)
    parser.add_argument('--prompt-dim', type=int, default=None)
    parser.add_argument('--mask-type', type=str, default='constant-0.1',
                        help='Missing-rate for the student, as constant-<rate>.')

    # ---- distillation: 4 clusters (CE, prob, proto, rel) -- see        ---- #
    # ---- train_student.py's module docstring. CE is fixed at weight   ---- #
    # ---- 1; prob/proto/rel are all tunable.                           ---- #
    parser.add_argument('--kd-temperature', type=float, default=2.0)
    parser.add_argument('--ot-epsilon', type=float, default=0.1, help='Sinkhorn entropic regularization.')
    parser.add_argument('--ot-iters', type=int, default=50, help='Sinkhorn iterations.')
    parser.add_argument('--lambda-prob', type=float, default=1.0,
                        help='Weight for the prob cluster (KD + OT). 0 disables it.')
    parser.add_argument('--lambda-proto', type=float, default=1.0,
                        help='Weight for the proto cluster (r vs class prototypes). 0 disables it.')
    parser.add_argument('--lambda-rel', type=float, default=1.0,
                        help="Weight for the rel cluster (e_proto + node_rel). 0 disables it.")

    # ---- training (free to differ from the teacher) ---- #
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.00005, help='Tuned via tune_lr_l2_meld.py (teacher-side; reused here).')
    parser.add_argument('--l2', type=float, default=0.0001, help='Tuned via tune_lr_l2_meld.py (teacher-side; reused here).')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks the real valid split.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer for why adamw may be worth testing.')

    args = parser.parse_args()

    args.dataset = 'MELD'
    args.num_folder = 1
    args.n_classes = 7
    args.n_speakers = 9
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    mask_rate = float(args.mask_type.split('-')[-1])

    cuda = torch.cuda.is_available() and not args.no_cuda
    set_seed(args.seed, cuda)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    save_log = os.path.join(path.LOG_DIR, 'student_meld')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f"{save_log}/{time_now}_student_meld.txt", stream=sys.stdout)

    print(args)

    print('\n========== Reading Data ==========\n')
    assert os.path.exists(path.PATH_MELD), 'PATH_MELD not found!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders_meld(
        features_path=path.PATH_MELD,
        batch_size=args.batch_size, num_workers=0, roberta_idx=args.roberta_idx
    )
    train_loader, val_loader, test_loader = train_loaders[0], val_loaders[0], test_loaders[0]

    print(f'\n========== Training Student, MELD (genuine train/valid/test split, '
          f'best checkpoint = best valid F1, mask_rate={mask_rate}, prompt_type={args.prompt_type}, '
          f'lambda_prob={args.lambda_prob}, lambda_proto={args.lambda_proto}, lambda_rel={args.lambda_rel} '
          f'(CE fixed=1)) ==========\n')

    start_time = time.time()

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
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        (train_acc, train_fscore, train_loss, train_ce, train_kd, train_proto, train_ot,
         train_e_proto, train_node_rel) = train_or_eval_student(
            args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
            ot_loss, ot_cost, node_rel_loss,
            train_loader, mask_rate=mask_rate, optimizer=optimizer, train=True)
        (val_acc, val_fscore, val_loss, val_ce, val_kd, val_proto, val_ot,
         val_e_proto, val_node_rel) = train_or_eval_student(
            args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
            ot_loss, ot_cost, node_rel_loss,
            val_loader, mask_rate=mask_rate, optimizer=None, train=False)

        print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; '
              f'train_loss:{train_loss} (ce={train_ce}, kd={train_kd}, proto={train_proto}, ot={train_ot}, '
              f'e_proto={train_e_proto}, node_rel={train_node_rel}); '
              f'val_fscore:{val_fscore:2.2%}; val_acc:{val_acc:2.2%}; '
              f'val_loss:{val_loss} (ce={val_ce}, kd={val_kd}, proto={val_proto}, ot={val_ot}, '
              f'e_proto={val_e_proto}, node_rel={val_node_rel})')

        if val_fscore > best_val_fscore + args.early_stop_delta:
            best_val_fscore = val_fscore
            best_val_acc = val_acc
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
        test_loader, mask_rate=mask_rate, optimizer=None, train=False)

    print(f'\n>>>>> Student MELD: best_val_f1={best_val_fscore:.4f}, best_val_acc={best_val_acc:.4f}, '
          f'test_f1={test_fscore:.4f}, test_acc={test_acc:.4f}, '
          f'duration={time.time()-start_time:.2f}s >>>>>')

    save_student_checkpoint(best_model_state, args.dataset, fold_index=0)

    print(f'\n====== Student summary (MELD): test F1={test_fscore:.4f}, test acc={test_acc:.4f} ======')
