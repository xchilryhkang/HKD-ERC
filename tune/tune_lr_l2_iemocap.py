import os
import sys
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
from src.utils import get_loaders, build_model, make_optimizer, set_seed
from train.train_teacher import train_or_eval_model
from src.losses.loss import MaskedCELoss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lrs', type=str, default='0.00005,0.0001,0.0002,0.0005',
                        help='Comma-separated learning rate candidates.')
    parser.add_argument('--l2s', type=str, default='0,0.00001,0.0001',
                        help='Comma-separated weight-decay (L2) candidates.')
    parser.add_argument('--seeds', type=str, default='197,649,722',
                        help='Comma-separated training seeds.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer.')

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
    parser.add_argument('--skip-combos', type=int, default=0,
                        help='Skip the first N (lr,l2) combos in iteration order (resume mid-sweep).')
    parser.add_argument('--append-to', type=str, default=None,
                        help='Existing log file to append to, instead of creating a new one (resume mid-sweep).')

    args = parser.parse_args()
    lrs = [float(x) for x in args.lrs.split(',')]
    l2s = [float(x) for x in args.l2s.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    args.mask_type = 'constant-0.0'
    args.prompt_type = 'none'
    args.n_proto = 10
    args.prompt_dim = None
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.num_folder, args.n_classes, args.n_speakers = (
        (5, 4, 2) if args.dataset == 'IEMOCAPFour' else (5, 6, 2)
    )
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available() and not args.no_cuda

    n_combos = len(lrs) * len(l2s)

    if args.append_to:
        result_path = args.append_to
        assert os.path.exists(result_path), f'--append-to path not found: {result_path}'
    else:
        out_dir = os.path.join(path.SAVED_ROOT, 'tune')
        os.makedirs(out_dir, exist_ok=True)
        time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        result_path = os.path.join(out_dir, f'{time_now}_lr_l2_sweep_iemocap_{args.optimizer}_{args.dataset}.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    if args.skip_combos:
        log(f'\n===== RESUMING from combo {args.skip_combos + 1}/{n_combos} (skipping first {args.skip_combos}) =====')
    else:
        log(f'lr x l2 tune (IEMOCAP 5-fold leave-one-session-out CV, best=session-out F1) — '
            f'dataset={args.dataset}  optimizer={args.optimizer}')
        log(f'lrs={lrs}  l2s={l2s}  ({n_combos} combos)  training_seeds={seeds}  '
            f'epochs_cap={args.epochs}  early_stop_patience={args.early_stop_patience}')
        log(f'lstm_hidden_size={args.lstm_hidden_size}  graph-layers={args.graph_layers}  '
            f'hg-layers={args.hg_layers}  n-graph-heads={args.n_graph_heads}  dropout={args.dropout}')
        log('-' * 100)

    print('\n========== Reading Data ==========\n')
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    train_loaders, _unused_val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
    )

    results = []
    combo_idx = 0

    for lr in lrs:
        for l2 in l2s:
            combo_idx += 1
            if combo_idx <= args.skip_combos:
                continue
            log(f'\n[{combo_idx}/{n_combos}] lr={lr} l2={l2}')

            seed_means = []
            combo_start = time.time()

            for seed in seeds:
                set_seed(seed, cuda)

                fold_f1 = []
                for ii in range(args.num_folder):
                    model = build_model(args, adim, tdim, vdim)
                    cls_loss = MaskedCELoss()
                    if cuda:
                        model.cuda()
                        cls_loss.cuda()
                    args.lr, args.l2 = lr, l2
                    optimizer = make_optimizer(model.parameters(), args)

                    best_f1, best_state, no_improve = 0.0, None, 0
                    for epoch in range(args.epochs):
                        train_or_eval_model(args, model, cls_loss, train_loaders[ii], mask_rate=0.0,
                                             optimizer=optimizer, train=True)
                        so_acc, so_f1, _ = train_or_eval_model(
                            args, model, cls_loss, test_loaders[ii], mask_rate=0.0, optimizer=None, train=False)
                        if so_f1 > best_f1 + args.early_stop_delta:
                            best_f1, no_improve = so_f1, 0
                            best_state = copy.deepcopy(model.state_dict())
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

            log(f'>>>>> lr={lr} l2={l2}: sessionout_f1 mean={mean_f1:.4f} std={std_f1:.4f} | '
                f'duration={duration:.1f}s >>>>>')

            results.append((lr, l2, mean_f1, std_f1, duration))

    log('\n===== SUMMARY (sorted by mean sessionout_f1, best first) =====')
    for lr, l2, mean_f1, std_f1, duration in sorted(results, key=lambda r: -r[2]):
        log(f'lr={lr:<10} l2={l2:<10}  sessionout_f1={mean_f1:.4f}±{std_f1:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
