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
from src.utils import get_loaders_meld, build_model, set_seed
from train.train_teacher import train_or_eval_model
from src.losses.loss import MaskedCELoss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--windows', type=str, default='1,2,3,4,5',
                        help='Comma-separated window candidates (windowp=windowf=w for each).')
    parser.add_argument('--lr', type=float, default=0.00005, help='Fixed lr (from tune_lr_l2_meld.py).')
    parser.add_argument('--l2', type=float, default=0.0001, help='Fixed l2 (from tune_lr_l2_meld.py).')
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='Fixed (from tune_hidden_dim_meld.py).')
    parser.add_argument('--seeds', type=str, default='197,649,722',
                        help='Comma-separated training seeds.')
    parser.add_argument('--roberta-idx', type=int, default=1, choices=[1, 2, 3, 4])

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- early stopping tracks the real valid split.')
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

    args = parser.parse_args()
    windows = [int(x) for x in args.windows.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

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
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available() and not args.no_cuda

    n_combos = len(windows)

    out_dir = os.path.join(path.SAVED_ROOT, 'tune')
    os.makedirs(out_dir, exist_ok=True)
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    result_path = os.path.join(out_dir, f'{time_now}_window_sweep_meld.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'window tune (MELD, genuine train/valid/test split)')
    log(f'windows={windows} (windowp=windowf=w)  ({n_combos} candidates)  training_seeds={seeds}  '
        f'lr={args.lr} (fixed)  l2={args.l2} (fixed)  lstm_hidden_size={args.lstm_hidden_size} (fixed)')
    log(f'graph-layers={args.graph_layers}  hg-layers={args.hg_layers}  n-graph-heads={args.n_graph_heads}  '
        f'dropout={args.dropout}  roberta_idx={args.roberta_idx}')
    log('-' * 100)

    print('\n========== Reading Data ==========\n')
    assert os.path.exists(path.PATH_MELD), 'PATH_MELD not found!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders_meld(
        features_path=path.PATH_MELD,
        batch_size=args.batch_size, num_workers=0, roberta_idx=args.roberta_idx
    )
    train_loader, val_loader, test_loader = train_loaders[0], val_loaders[0], test_loaders[0]

    results = []

    for w in windows:
        args.windowp = w
        args.windowf = w
        log(f'\n[windowp=windowf={w}]')

        seed_val, seed_test = [], []
        combo_start = time.time()

        for seed in seeds:
            set_seed(seed, cuda)

            model = build_model(args, adim, tdim, vdim)
            cls_loss = MaskedCELoss()
            if cuda:
                model.cuda()
                cls_loss.cuda()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            best_val_f1, best_model_state, no_improve = 0.0, None, 0
            for epoch in range(args.epochs):
                train_or_eval_model(args, model, cls_loss, train_loader, mask_rate=0.0,
                                     optimizer=optimizer, train=True)
                val_acc, val_f1, _ = train_or_eval_model(
                    args, model, cls_loss, val_loader, mask_rate=0.0, optimizer=None, train=False)
                if val_f1 > best_val_f1 + args.early_stop_delta:
                    best_val_f1, no_improve = val_f1, 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    no_improve += 1
                    if no_improve >= args.early_stop_patience:
                        break

            model.load_state_dict(best_model_state)
            test_acc, test_f1, _ = train_or_eval_model(
                args, model, cls_loss, test_loader, mask_rate=0.0, optimizer=None, train=False)

            log(f'    seed={seed} val_f1={best_val_f1:.4f} test_f1={test_f1:.4f}')
            seed_val.append(best_val_f1)
            seed_test.append(test_f1)

        mean_val, std_val = float(np.mean(seed_val)), float(np.std(seed_val))
        mean_test, std_test = float(np.mean(seed_test)), float(np.std(seed_test))
        duration = time.time() - combo_start

        log(f'>>>>> window={w}: val_f1 mean={mean_val:.4f} std={std_val:.4f} | '
            f'test_f1 mean={mean_test:.4f} std={std_test:.4f} | duration={duration:.1f}s >>>>>')

        results.append((w, mean_val, std_val, mean_test, std_test, duration))

    log('\n===== SUMMARY (sorted by mean val_f1, best first) =====')
    for w, mean_val, std_val, mean_test, std_test, duration in sorted(results, key=lambda r: -r[1]):
        log(f'window={w:<4}  val_f1={mean_val:.4f}±{std_val:.4f}  '
            f'test_f1={mean_test:.4f}±{std_test:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
