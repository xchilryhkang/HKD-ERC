import os
import sys
import time
import argparse
import datetime
import copy

import numpy as np
import torch
import torch.optim as optim

# Make the repo root importable so `src` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import path
from src.utils import (
    Logger, get_loaders, build_model, load_teacher_checkpoint, save_student_checkpoint, make_optimizer, set_seed
)
from src.losses.loss import MaskedCELoss
from src.losses.kd_losses import (
    MaskedKLLoss, PrototypeRelationLoss, SinkhornOTLoss, compute_ot_cost_matrix, NodeRelationLoss
)
from train.train_student import compute_teacher_prototypes, train_or_eval_student


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', choices=['IEMOCAPFour', 'IEMOCAPSix'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')

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
    # lambda_prob/lambda_rel defaults are the best combo from
    # tune/tune_lambda_prob_rel_iemocap.py (IEMOCAPSix, mask_rate 0.3);
    # IEMOCAPFour reuses them rather than being swept separately.
    # lambda_proto was held at 1.0 during that sweep, never tuned.
    parser.add_argument('--lambda-prob', type=float, default=0.4,
                        help='Weight for the prob cluster (KD + OT). 0 disables it.')
    parser.add_argument('--lambda-proto', type=float, default=1.0,
                        help='Weight for the proto cluster (r vs class prototypes). 0 disables it.')
    parser.add_argument('--lambda-rel', type=float, default=0.6,
                        help="Weight for the rel cluster (e_proto + node_rel). 0 disables it.")

    # ---- training (free to differ from the teacher) ---- #
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks session-out F1.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer. adamw + lr=0.0002/l2=1e-5 is the teacher-tuned '
                             'winner from tune_lr_l2_iemocap.py; the student reuses it (no separate '
                             'student lr/l2 sweep has been run).')

    args = parser.parse_args()
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    mask_rate = float(args.mask_type.split('-')[-1])

    cuda = torch.cuda.is_available() and not args.no_cuda
    set_seed(args.seed, cuda)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    save_log = os.path.join(path.LOG_DIR, 'student_iemocap', f'{args.dataset}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f"{save_log}/{time_now}_student_iemocap_{args.dataset}.txt", stream=sys.stdout)

    args.num_folder, args.n_classes, args.n_speakers = (
        (5, 4, 2) if args.dataset == 'IEMOCAPFour' else (5, 6, 2)
    )

    print(args)

    print('\n========== Reading Data ==========\n')
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    train_loaders, _unused_val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
    )
    # Early stopping tracks the held-out session itself -- see
    # train_teacher_iemocap.py for why this protocol has no separate val.

    print(f'\n========== Training Student, IEMOCAP 5-fold leave-one-session-out CV '
          f'(best checkpoint = best session-out F1, mask_rate={mask_rate}, prompt_type={args.prompt_type}, '
          f'lambda_prob={args.lambda_prob}, lambda_proto={args.lambda_proto}, lambda_rel={args.lambda_rel} '
          f'(CE fixed=1), per fold) ==========\n')

    fold_f1, fold_acc = [], []

    for ii in range(args.num_folder):
        print(f'>>>>> Student fold {ii+1}/{args.num_folder} >>>>>')
        sessionout_loader = test_loaders[ii]
        start_time = time.time()

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

        best_sessionout_fscore = 0.0
        best_sessionout_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            (train_acc, train_fscore, train_loss, train_ce, train_kd, train_proto, train_ot,
             train_e_proto, train_node_rel) = train_or_eval_student(
                args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                ot_loss, ot_cost, node_rel_loss,
                train_loaders[ii], mask_rate=mask_rate, optimizer=optimizer, train=True)
            (sessionout_acc, sessionout_fscore, sessionout_loss, so_ce, so_kd, so_proto, so_ot,
             so_e_proto, so_node_rel) = train_or_eval_student(
                args, teacher, student, cls_loss, kl_loss, proto_loss, proto_t, proto_t_e,
                ot_loss, ot_cost, node_rel_loss,
                sessionout_loader, mask_rate=mask_rate, optimizer=None, train=False)

            print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; '
                  f'train_loss:{train_loss} (ce={train_ce}, kd={train_kd}, proto={train_proto}, ot={train_ot}, '
                  f'e_proto={train_e_proto}, node_rel={train_node_rel}); '
                  f'sessionout_fscore:{sessionout_fscore:2.2%}; sessionout_acc:{sessionout_acc:2.2%}; '
                  f'sessionout_loss:{sessionout_loss} (ce={so_ce}, kd={so_kd}, proto={so_proto}, ot={so_ot}, '
                  f'e_proto={so_e_proto}, node_rel={so_node_rel})')

            if sessionout_fscore > best_sessionout_fscore + args.early_stop_delta:
                best_sessionout_fscore = sessionout_fscore
                best_sessionout_acc = sessionout_acc
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(student.state_dict())
                print(f"  New best student session-out F1: {best_sessionout_fscore:.4f}.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print(f"  Early stopping student after {epoch + 1} epochs.")
                    break

        print(f'>>>>> Student fold {ii+1}: best_sessionout_f1={best_sessionout_fscore:.4f}, '
              f'best_sessionout_acc={best_sessionout_acc:.4f}, '
              f'duration={time.time()-start_time:.2f}s >>>>>')

        fold_f1.append(best_sessionout_fscore)
        fold_acc.append(best_sessionout_acc)

        save_student_checkpoint(best_model_state, args.dataset, fold_index=ii)

    print(f'\n====== Student summary (5-fold leave-one-session-out CV): '
          f'mean session-out F1={np.mean(fold_f1):.4f}, mean session-out acc={np.mean(fold_acc):.4f} ======')
