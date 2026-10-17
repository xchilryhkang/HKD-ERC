"""
Isolated windowp/windowf sweep, run on top of the best hidden_dim/lr/l2/
graph_layers/hg_layers/n_graph_heads found so far (hidden_dim=256,
lr=0.0001, l2=1e-5, graph_layers=1, hg_layers=1, n_graph_heads=4). NOT
part of the main pipeline — only imports from utils.py / train_teacher.py
/ loss.py, and nothing imports IT back, so deleting this one file later
removes the sweep cleanly with no side effects anywhere else.

Loops a single window size w (default 1,2,3,4,5), setting
windowp = windowf = w for each — i.e. past and future context are kept
symmetric (see context_graph_module.md's ContextGraphLayer window
definition), not swept independently. Reuses train_teacher.py's exact
train_or_eval_model loop for each candidate — same masking, same early
stopping, same as a real train_teacher.py run with matching
--windowp/--windowf.

hidden_dim, lr, l2, graph_layers, hg_layers, n_graph_heads are all held
FIXED (CLI-overridable, but not part of this sweep) — this is a 1D sweep
over window size only.

Note: windowp/windowf do NOT affect any parameter shape (they only change
how many edges batch_graphify builds at runtime), so unlike hidden_dim/
graph_layers/hg_layers/n_graph_heads this choice is free to differ later
between teacher and student without breaking checkpoint compatibility.

Data is loaded ONCE and reused across every candidate (loading is the slow
part) — only model re-init + training repeats per candidate.

By default every fold is trained per candidate (same as train_teacher.py —
5 for IEMOCAP, 1 for MELD) and the reported val/test F1 is the mean across
folds. Pass --num-folds-to-run 1 for a faster, cheaper screening pass
instead (results then won't be directly comparable to a full run).

Results are written to their own .txt file under output/tune/ (separate
from the other tune_*.py scripts' logs and train_teacher.py's normal
log), plus printed to the console.
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
from train_teacher import train_or_eval_model
from loss import MaskedCELoss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--windows', type=str, default='1,2,3,4,5',
                        help='Comma-separated window sizes w to sweep; each sets windowp=windowf=w.')

    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Fixed hidden_dim for every candidate (-> lstm_hidden_size = this // 2).')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--graph-layers', type=int, default=1)
    parser.add_argument('--hg-layers', type=int, default=1)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--num-folds-to-run', type=int, default=None,
                        help='Folds (starting at fold 0) trained per candidate, then averaged. '
                             'Defaults to the dataset\'s full fold count (5 for IEMOCAP, 1 for MELD). '
                             'Pass a smaller number (e.g. 1) for a faster screening pass instead.')

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', help='IEMOCAPFour, IEMOCAPSix or MELD')
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=50)

    args = parser.parse_args()
    windows = [int(x) for x in args.windows.split(',')]

    assert args.hidden_dim % 2 == 0, f'hidden_dim {args.hidden_dim} must be even'
    assert args.hidden_dim % args.n_graph_heads == 0, (
        f'hidden_dim {args.hidden_dim} must be divisible by n_graph_heads ({args.n_graph_heads})'
    )
    args.lstm_hidden_size = args.hidden_dim // 2

    # Same as train_teacher.py: teacher sees every modality, no placeholder.
    args.mask_type = 'constant-0.0'
    args.prompt_type = 'none'
    args.n_proto = 10
    args.prompt_dim = None
    args.d_e = None
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available() and not args.no_cuda
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'IEMOCAPFour':
        args.num_folder, args.n_classes, args.n_speakers = 5, 4, 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder, args.n_classes, args.n_speakers = 5, 6, 2
    elif args.dataset == 'MELD':
        args.num_folder, args.n_classes, args.n_speakers = 1, 7, 9
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    num_folds_to_run = args.num_folder if args.num_folds_to_run is None else min(args.num_folds_to_run, args.num_folder)
    n_combos = len(windows)

    out_dir = os.path.join(path.SAVED_ROOT, 'tune')
    os.makedirs(out_dir, exist_ok=True)
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    result_path = os.path.join(out_dir, f'{time_now}_window_sweep_{args.dataset}.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'window sweep (windowp=windowf=w) — dataset={args.dataset}')
    log(f'windows={windows}  ({n_combos} candidates)')
    log(f'hidden_dim={args.hidden_dim} (lstm_hidden_size={args.lstm_hidden_size}, fixed)  '
        f'lr={args.lr} (fixed)  l2={args.l2} (fixed)  graph_layers={args.graph_layers} (fixed)  '
        f'hg_layers={args.hg_layers} (fixed)  n_graph_heads={args.n_graph_heads} (fixed)')
    log(f'dropout={args.dropout}  batch_size={args.batch_size}  '
        f'folds_per_candidate={num_folds_to_run}/{args.num_folder}  seed={args.seed}')
    log('-' * 90)

    print('\n========== Reading Data ==========\n')
    if args.dataset == 'MELD':
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=0, meld_path=path.PATH_MELD
        )
    else:
        audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
        text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
        video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
        assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), 'features not exist!'
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            audio_root=audio_root, text_root=text_root, video_root=video_root,
            num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
        )

    results = []

    for idx, w in enumerate(windows):
        args.windowp = w
        args.windowf = w

        log(f'[{idx + 1}/{n_combos}] windowp=windowf={w}')

        fold_val_f1, fold_test_f1, fold_test_acc = [], [], []
        start_time = time.time()

        for ii in range(num_folds_to_run):
            model = build_model(args, adim, tdim, vdim)
            cls_loss = MaskedCELoss()
            if cuda:
                model.cuda()
                cls_loss.cuda()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            best_val_fscore = 0.0
            best_model_state = None
            epochs_no_improve = 0
            epoch = 0

            for epoch in range(args.epochs):
                train_acc, train_fscore, train_loss = train_or_eval_model(
                    args, model, cls_loss, train_loaders[ii], mask_rate=0.0, optimizer=optimizer, train=True)
                val_acc, val_fscore, val_loss = train_or_eval_model(
                    args, model, cls_loss, val_loaders[ii], mask_rate=0.0, optimizer=None, train=False)

                if val_fscore > best_val_fscore + args.early_stop_delta:
                    best_val_fscore = val_fscore
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.early_stop_patience:
                        break

            model.load_state_dict(best_model_state)
            test_acc, test_fscore, test_loss = train_or_eval_model(
                args, model, cls_loss, test_loaders[ii], mask_rate=0.0, optimizer=None, train=False)

            log(f'  fold={ii} epochs_run={epoch + 1} best_val_f1={best_val_fscore:.4f} '
                f'test_f1={test_fscore:.4f} test_acc={test_acc:.4f}')

            fold_val_f1.append(best_val_fscore)
            fold_test_f1.append(test_fscore)
            fold_test_acc.append(test_acc)

        duration = time.time() - start_time
        mean_val_f1 = float(np.mean(fold_val_f1))
        mean_test_f1 = float(np.mean(fold_test_f1))
        mean_test_acc = float(np.mean(fold_test_acc))

        log(f'windowp=windowf={w}: mean_val_f1={mean_val_f1:.4f} mean_test_f1={mean_test_f1:.4f} '
            f'mean_test_acc={mean_test_acc:.4f} duration={duration:.1f}s')
        log('-' * 90)

        results.append((w, mean_val_f1, mean_test_f1, mean_test_acc, duration))

    log('\n===== SUMMARY (sorted by mean_val_f1, best first) =====')
    for w, mean_val_f1, mean_test_f1, mean_test_acc, duration in sorted(results, key=lambda r: -r[1]):
        log(f'window={w:2d}  val_f1={mean_val_f1:.4f}  test_f1={mean_test_f1:.4f}  '
            f'test_acc={mean_test_acc:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
