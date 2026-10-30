import os
import sys
import time
import argparse
import datetime
import copy

import numpy as np
import torch

# Make the repo root importable so `src` / `train` resolve from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import path
from src.utils import (
    Logger, get_loaders_cmu, build_model, load_teacher_checkpoint,
    save_student_checkpoint, make_optimizer, set_seed, PROTO_BIN_CHOICES, PROTO_BIN_MODES
)
from src.losses.loss import MaskedMSELoss
from src.losses.kd_losses import PrototypeRelationLoss, NodeRelationLoss
from train.train_student import (compute_teacher_prototypes, train_or_eval_student,
                                 calibrate_loss_scales)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CMUMOSI', choices=['CMUMOSI', 'CMUMOSEI'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')

    # ---- architecture: MUST match the teacher checkpoint ---- #
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

    # ---- distillation: same 4 clusters as the classification datasets, with ---- #
    # ---- the two label-dependent ones adapted to regression -- see below.   ---- #
    parser.add_argument('--lambda-prob', type=float, default=1.0,
                        help='Weight for the prob cluster. Under regression this is MSE between '
                             'the student and teacher scalars (KL/OT need class distributions); '
                             '0 disables it.')
    parser.add_argument('--lambda-proto', type=float, default=1.0,
                        help='Weight for the proto cluster (r vs the binned-sentiment '
                             'prototypes, see --proto-bins). 0 disables it.')
    parser.add_argument('--lambda-rel', type=float, default=1.0,
                        help='Weight for the rel cluster (e_proto + node_rel). 0 disables it.')
    parser.add_argument('--proto-bins', type=int, default=None, choices=PROTO_BIN_CHOICES,
                        help='How many pseudo-classes the continuous score is binned into for '
                             'the proto/rel clusters (see utils.bin_sentiment). An even count '
                             'puts 0 on a bin edge so no prototype mixes polarities (and the '
                             'exact zeros are dropped); an odd count centres a bin on 0. '
                             'Defaults to 6 on CMUMOSEI and 4 on CMUMOSI, whose 1284 train '
                             'utterances leave the smallest bin at 67 by 6 but 218 by 4.')
    parser.add_argument('--proto-bin-mode', type=str, default='uniform', choices=PROTO_BIN_MODES,
                        help="'uniform' cuts [-3, 3] into equal segments; 'round' reproduces the "
                             'Acc-7/Acc-5 binning, whose clipped extreme bins are half-width and '
                             'much thinner (24 vs 67 on MOSI at 7 bins) and which only supports '
                             'an odd --proto-bins. Ignored at 2 bins.')

    # ---- training ---- #
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='Carried over from MELD; NOT tuned on CMU yet.')
    parser.add_argument('--l2', type=float, default=0.0001,
                        help='Carried over from MELD; NOT tuned on CMU yet.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Upper bound -- early stopping tracks the real valid split.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--loss-norm', type=str, default='none', choices=['none', 'calibrated'],
                        help="'calibrated' rescales prob, proto and rel to the MSE-slot scale "
                             'once per run, so their lambdas are meaningful in [0, 1]. Unlike '
                             'the ERC scripts this covers prob too, which here is MSE against '
                             'the teacher scalar and runs at 0.039x (MOSI) to 0.167x (MOSEI) of '
                             'the gold MSE. See train_student.calibrate_loss_scales.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])

    args = parser.parse_args()

    # Regression setup, matching train_teacher_cmu.py. The one extra knob the
    # student needs is proto_bins: the prototype and rel clusters are
    # per-class, so the continuous score is binned into that many
    # pseudo-classes (see utils.bin_sentiment). The metrics are unaffected --
    # they always use sign with zeros dropped, as reported for these datasets.
    args.task = 'regression'
    args.num_folder = 1
    args.n_classes = 1
    if args.proto_bins is None:
        args.proto_bins = 4 if args.dataset == 'CMUMOSI' else 6
    args.n_proto_classes = args.proto_bins
    args.n_speakers = 1
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    mask_rate = float(args.mask_type.split('-')[-1])

    cuda = torch.cuda.is_available() and not args.no_cuda
    set_seed(args.seed, cuda)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    save_log = os.path.join(path.LOG_DIR, 'student_cmu', args.dataset)
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f'{save_log}/{time_now}_student_{args.dataset}.txt', stream=sys.stdout)

    print(args)

    print('\n========== Reading Data ==========\n')
    feat_root = path.PATH_TO_FEATURES_Win[args.dataset]
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders_cmu(
        dataset=args.dataset,
        batch_size=args.batch_size,
        audio_root=os.path.join(feat_root, args.audio_feature),
        text_root=os.path.join(feat_root, args.text_feature),
        video_root=os.path.join(feat_root, args.video_feature),
        num_workers=0,
    )
    train_loader, val_loader, test_loader = train_loaders[0], val_loaders[0], test_loaders[0]

    print(f'\n========== Training Student, {args.dataset} (sentiment regression, '
          f'best checkpoint = best valid F1, mask_rate={mask_rate}, '
          f'prompt_type={args.prompt_type}, lambda_prob={args.lambda_prob}, '
          f'lambda_proto={args.lambda_proto}, lambda_rel={args.lambda_rel} '
          f'(MSE fixed=1)) ==========\n')

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
    print(f'  teacher prototypes computed: shape={tuple(proto_t.shape)}, '
          f'proto_t_e shape={tuple(proto_t_e.shape)} '
          f'({args.proto_bins} {args.proto_bin_mode} sentiment bins)')

    student = build_model(args, adim, tdim, vdim)
    if cuda:
        student.cuda()

    # The CE slot is MSE here; kl_loss / ot_loss / ot_cost are never reached
    # on the regression path (see train_student.train_or_eval_student), so
    # they are passed as None rather than constructed.
    cls_loss = MaskedMSELoss()
    proto_loss = PrototypeRelationLoss()
    node_rel_loss = NodeRelationLoss()
    if cuda:
        cls_loss.cuda()

    args.prob_scale, args.proto_scale, args.rel_scale = 1.0, 1.0, 1.0
    if args.loss_norm == 'calibrated':
        # with_prob: unlike the ERC datasets, prob here is not already
        # CE-scale (see train_student.calibrate_loss_scales).
        args.prob_scale, args.proto_scale, args.rel_scale = calibrate_loss_scales(
            args, teacher, student, cls_loss, None, proto_loss, proto_t, proto_t_e,
            None, None, node_rel_loss, train_loader, mask_rate=mask_rate, with_prob=True)
        print(f'  loss-norm calibrated: prob x{args.prob_scale:.0f}, '
              f'proto x{args.proto_scale:.0f}, rel x{args.rel_scale:.0f}')

    optimizer = make_optimizer(student.parameters(), args)

    best_val_fscore = 0.0
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        (train_acc, train_fscore, train_loss, train_ce, train_kd, train_proto, train_ot,
         train_e_proto, train_node_rel) = train_or_eval_student(
            args, teacher, student, cls_loss, None, proto_loss, proto_t, proto_t_e,
            None, None, node_rel_loss,
            train_loader, mask_rate=mask_rate, optimizer=optimizer, train=True)
        (val_acc, val_fscore, val_loss, val_ce, val_kd, val_proto, val_ot,
         val_e_proto, val_node_rel, val_extra) = train_or_eval_student(
            args, teacher, student, cls_loss, None, proto_loss, proto_t, proto_t_e,
            None, None, node_rel_loss,
            val_loader, mask_rate=mask_rate, optimizer=None, train=False, return_extra=True)

        print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; '
              f'train_loss:{train_loss} (mse={train_ce}, kd={train_kd}, proto={train_proto}, '
              f'e_proto={train_e_proto}, node_rel={train_node_rel}); '
              f'val_fscore:{val_fscore:2.2%}; val_acc:{val_acc:2.2%}; '
              f'val_loss:{val_loss} (mse={val_ce}, kd={val_kd}, proto={val_proto}, '
              f'e_proto={val_e_proto}, node_rel={val_node_rel}); '
              f'val_mae:{val_extra["mae"]:.4f}; val_corr:{val_extra["corr"]:.4f}')

        if val_fscore > best_val_fscore + args.early_stop_delta:
            best_val_fscore = val_fscore
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(student.state_dict())
            print(f'  New best student val F1: {best_val_fscore:.4f}.')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f'  Early stopping student after {epoch + 1} epochs.')
                break

    student.load_state_dict(best_model_state)
    (test_acc, test_fscore, test_loss, test_ce, test_kd, test_proto, test_ot,
     test_e_proto, test_node_rel, test_extra) = train_or_eval_student(
        args, teacher, student, cls_loss, None, proto_loss, proto_t, proto_t_e,
        None, None, node_rel_loss,
        test_loader, mask_rate=mask_rate, optimizer=None, train=False, return_extra=True)

    print(f'\n>>>>> Student {args.dataset}: best_val_f1={best_val_fscore:.4f}, '
          f'best_val_acc={best_val_acc:.4f}, test_f1={test_fscore:.4f}, test_acc={test_acc:.4f}, '
          f'test_mae={test_extra["mae"]:.4f}, test_corr={test_extra["corr"]:.4f}, '
          f'duration={time.time()-start_time:.2f}s >>>>>')

    save_student_checkpoint(best_model_state, args.dataset, fold_index=0)

    print(f'\n====== Student summary ({args.dataset}): test F1={test_fscore:.4f}, '
          f'test acc={test_acc:.4f}, MAE={test_extra["mae"]:.4f}, '
          f'Corr={test_extra["corr"]:.4f} (acc/F1 exclude score==0) ======')
