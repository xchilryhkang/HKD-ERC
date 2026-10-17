"""
Trains the teacher for IEMOCAP (Four/Six) using the standard 5-fold
leave-one-session-out CV protocol: for each fold, train on the 4 other
sessions, track performance on the held-out session EVERY epoch, and keep
the checkpoint from whichever epoch scored best on it (early stopping on
that signal). The 5-fold mean of each fold's best session-out F1/acc is
the final reported metric.

Why this is a separate file from train_teacher.py: train_teacher.py's
generic loop is written around train_loaders/val_loaders/test_loaders as
three conceptually separate things, which fits MELD (which really does
have three separate splits) but is misleading for IEMOCAP -- under this
CV protocol there IS NO separate validation session; what the loop
tracks every epoch simply IS the held-out session's own performance
(utils.get_loaders already returns val_loaders == test_loaders for
IEMOCAP precisely because there's nothing else to hand it). Naming a
local variable "val_fscore" there obscures that it's actually the
session-out score. This file names it for what it is instead:
sessionout_fscore, tracked directly off test_loaders, no val_loaders
involved anywhere.

MELD has a real, separate validation split and keeps using
train_teacher.py as-is.

Isolated and deletable: reuses train_or_eval_model/build_model/etc. from
train_teacher.py and utils.py, but nothing imports this file back, so it
can be deleted without side effects. Saves checkpoints via the SAME
save_teacher_checkpoint(dataset, fold_index) path train_teacher.py uses --
running this will overwrite that dataset's existing teacher checkpoints.
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
from erc.utils import Logger, get_loaders, build_model, save_teacher_checkpoint, make_optimizer, set_seed
from erc.losses.loss import MaskedCELoss
from train.train_teacher import train_or_eval_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', choices=['IEMOCAPFour', 'IEMOCAPSix'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks session-out F1.')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--d-e', type=int, default=None)
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer. adamw + lr=0.0002/l2=1e-5 is the current '
                             'best combo from tune_lr_l2_iemocap.py (session-out F1=0.6431±0.0011).')

    args = parser.parse_args()

    args.mask_type = 'constant-0.0'
    args.prompt_type = 'none'
    args.n_proto = 10
    args.prompt_dim = None
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate

    print(f"Setting random seed to: {args.seed}")
    cuda = torch.cuda.is_available() and not args.no_cuda
    set_seed(args.seed, cuda)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    save_log = os.path.join(path.LOG_DIR, 'teacher_iemocap', f'{args.dataset}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f"{save_log}/{time_now}_teacher_iemocap_{args.dataset}.txt", stream=sys.stdout)

    args.num_folder, args.n_classes, args.n_speakers = (
        (5, 4, 2) if args.dataset == 'IEMOCAPFour' else (5, 6, 2)
    )

    print(args)

    print('\n========== Reading Data ==========\n')
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), 'features not exist!'
    train_loaders, _unused_val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
    )
    # _unused_val_loaders IS test_loaders (see utils.get_loaders) -- never
    # touched below. There is no separate validation session under this CV
    # protocol; what we track every epoch below (sessionout_fscore) IS the
    # held-out session's own score, tracked honestly as such.

    print(f'\n========== Training Teacher, IEMOCAP 5-fold leave-one-session-out CV '
          f'(best checkpoint = best session-out F1, per fold) ==========\n')

    fold_f1, fold_acc = [], []

    for ii in range(args.num_folder):
        print(f'>>>>> Teacher fold {ii+1}/{args.num_folder} >>>>>')
        train_loader = train_loaders[ii]
        sessionout_loader = test_loaders[ii]
        start_time = time.time()

        model = build_model(args, adim, tdim, vdim)
        cls_loss = MaskedCELoss()
        if cuda:
            model.cuda()
            cls_loss.cuda()

        optimizer = make_optimizer(model.parameters(), args)

        best_sessionout_fscore = 0.0
        best_sessionout_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            train_acc, train_fscore, train_loss = train_or_eval_model(
                args, model, cls_loss, train_loader, mask_rate=0.0, optimizer=optimizer, train=True)
            sessionout_acc, sessionout_fscore, sessionout_loss = train_or_eval_model(
                args, model, cls_loss, sessionout_loader, mask_rate=0.0, optimizer=None, train=False)

            print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; '
                  f'train_loss:{train_loss}; sessionout_fscore:{sessionout_fscore:2.2%}; '
                  f'sessionout_acc:{sessionout_acc:2.2%}; sessionout_loss:{sessionout_loss}')

            if sessionout_fscore > best_sessionout_fscore + args.early_stop_delta:
                best_sessionout_fscore = sessionout_fscore
                best_sessionout_acc = sessionout_acc
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  New best teacher session-out F1: {best_sessionout_fscore:.4f}.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print(f"  Early stopping teacher after {epoch + 1} epochs.")
                    break

        print(f'>>>>> Teacher fold {ii+1}: best_sessionout_f1={best_sessionout_fscore:.4f}, '
              f'best_sessionout_acc={best_sessionout_acc:.4f}, '
              f'duration={time.time()-start_time:.2f}s >>>>>')

        fold_f1.append(best_sessionout_fscore)
        fold_acc.append(best_sessionout_acc)

        save_teacher_checkpoint(best_model_state, args.dataset, fold_index=ii)

    print(f'\n====== Teacher summary (5-fold leave-one-session-out CV): '
          f'mean session-out F1={np.mean(fold_f1):.4f}, mean session-out acc={np.mean(fold_acc):.4f} ======')
