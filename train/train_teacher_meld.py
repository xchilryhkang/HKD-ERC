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
from src.utils import Logger, get_loaders_meld, build_model, save_teacher_checkpoint, make_optimizer, set_seed
from src.losses.loss import MaskedCELoss
from train.train_teacher import train_or_eval_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--roberta-idx', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Which of the 4 RoBERTa text variants to use (see dataloader_meld.MELDRobertaDataset).')
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.00005, help='Tuned via tune_lr_l2_meld.py.')
    parser.add_argument('--l2', type=float, default=0.0001, help='Tuned via tune_lr_l2_meld.py.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks the real valid split.')
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
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer for why adamw may be worth testing.')

    args = parser.parse_args()

    args.dataset = 'MELD'
    args.num_folder = 1
    args.n_classes = 7
    args.n_speakers = 9
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

    save_log = os.path.join(path.LOG_DIR, 'teacher_meld')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f"{save_log}/{time_now}_teacher_meld.txt", stream=sys.stdout)

    print(args)

    print('\n========== Reading Data ==========\n')
    assert os.path.exists(path.PATH_MELD), 'PATH_MELD not found!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders_meld(
        features_path=path.PATH_MELD,
        batch_size=args.batch_size, num_workers=0, roberta_idx=args.roberta_idx
    )

    print(f'\n========== Training Teacher, MELD (genuine train/valid/test split, '
          f'best checkpoint = best valid F1) ==========\n')

    train_loader, val_loader, test_loader = train_loaders[0], val_loaders[0], test_loaders[0]
    start_time = time.time()

    model = build_model(args, adim, tdim, vdim)
    cls_loss = MaskedCELoss()
    if cuda:
        model.cuda()
        cls_loss.cuda()

    optimizer = make_optimizer(model.parameters(), args)

    best_val_fscore = 0.0
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_acc, train_fscore, train_loss = train_or_eval_model(
            args, model, cls_loss, train_loader, mask_rate=0.0, optimizer=optimizer, train=True)
        val_acc, val_fscore, val_loss = train_or_eval_model(
            args, model, cls_loss, val_loader, mask_rate=0.0, optimizer=None, train=False)

        print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; '
              f'train_loss:{train_loss}; val_fscore:{val_fscore:2.2%}; val_acc:{val_acc:2.2%}; '
              f'val_loss:{val_loss}')

        if val_fscore > best_val_fscore + args.early_stop_delta:
            best_val_fscore = val_fscore
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best teacher val F1: {best_val_fscore:.4f}.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"  Early stopping teacher after {epoch + 1} epochs.")
                break

    model.load_state_dict(best_model_state)
    test_acc, test_fscore, test_loss = train_or_eval_model(
        args, model, cls_loss, test_loader, mask_rate=0.0, optimizer=None, train=False)

    print(f'\n>>>>> Teacher MELD: best_val_f1={best_val_fscore:.4f}, best_val_acc={best_val_acc:.4f}, '
          f'test_f1={test_fscore:.4f}, test_acc={test_acc:.4f}, '
          f'duration={time.time()-start_time:.2f}s >>>>>')

    save_teacher_checkpoint(best_model_state, args.dataset, fold_index=0)

    print(f'\n====== Teacher summary (MELD): test F1={test_fscore:.4f}, test acc={test_acc:.4f} ======')
