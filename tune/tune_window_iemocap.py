"""
Tunes windowp/windowf (symmetric, windowp=windowf=w) for the teacher on
IEMOCAPSix, standard leave-one-session-out protocol (see
train_teacher_iemocap.py's module docstring): all 4 train sessions used,
fixed epochs, no validation split. lr/l2 are held FIXED at the values
tune_lr_l2_iemocap.py found best (lr=0.0001, l2=0.0001). Repeated across
multiple seeds to report mean/std of the 5-fold CV test F1 per window
value.

Isolated and deletable: only imports from train_teacher.py / utils.py /
loss.py, and nothing imports this file back. Results are written to their
own .txt file under output/tune/, plus printed to the console.
"""

import os
import sys
import time
import argparse
import datetime

import numpy as np
import torch
import torch.optim as optim

# Make the repo root importable so `erc` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erc import path
from erc.utils import get_loaders, build_model
from train.train_teacher import train_or_eval_model
from erc.losses.loss import MaskedCELoss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--windows', type=str, default='1,2,3,4,5',
                        help='Comma-separated window candidates (windowp=windowf=w for each).')
    parser.add_argument('--lr', type=float, default=0.0001, help='Fixed lr (from tune_lr_l2_iemocap.py).')
    parser.add_argument('--l2', type=float, default=0.0001, help='Fixed l2 (from tune_lr_l2_iemocap.py).')
    parser.add_argument('--seeds', type=str, default='197,649,722',
                        help='Comma-separated training seeds.')
    parser.add_argument('--epochs', type=int, default=40, help='Fixed epoch count per fold (no early stopping).')

    parser.add_argument('--dataset', type=str, default='IEMOCAPSix', choices=['IEMOCAPFour', 'IEMOCAPSix'])
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
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

    n_combos = len(windows)

    out_dir = os.path.join(path.SAVED_ROOT, 'tune')
    os.makedirs(out_dir, exist_ok=True)
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    result_path = os.path.join(out_dir, f'{time_now}_window_sweep_iemocap_noval_{args.dataset}.txt')

    def log(msg):
        print(msg)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'window re-tune (standard LOSO-CV, no val, fixed epochs) — dataset={args.dataset}')
    log(f'windows={windows} (windowp=windowf=w)  ({n_combos} combos)  training_seeds={seeds}  epochs={args.epochs}')
    log(f'lr={args.lr} (fixed)  l2={args.l2} (fixed)  lstm_hidden_size={args.lstm_hidden_size}  '
        f'graph-layers={args.graph_layers}  hg-layers={args.hg_layers}  n-graph-heads={args.n_graph_heads}  '
        f'dropout={args.dropout}')
    log('-' * 100)

    print('\n========== Reading Data ==========\n')
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), 'features not exist!'
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
    )

    results = []
    combo_idx = 0

    for w in windows:
        combo_idx += 1
        args.windowp = w
        args.windowf = w
        log(f'\n[{combo_idx}/{n_combos}] windowp=windowf={w}')

        seed_test_means = []
        combo_start = time.time()

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            fold_test_f1 = []
            for ii in range(args.num_folder):
                model = build_model(args, adim, tdim, vdim)
                cls_loss = MaskedCELoss()
                if cuda:
                    model.cuda()
                    cls_loss.cuda()
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

                for epoch in range(args.epochs):
                    train_or_eval_model(args, model, cls_loss, train_loaders[ii], mask_rate=0.0,
                                         optimizer=optimizer, train=True)

                test_acc, test_fscore, test_loss = train_or_eval_model(
                    args, model, cls_loss, test_loaders[ii], mask_rate=0.0, optimizer=None, train=False)

                log(f'    seed={seed} fold={ii} test_f1={test_fscore:.4f}')
                fold_test_f1.append(test_fscore)

            seed_test_mean = float(np.mean(fold_test_f1))
            seed_test_means.append(seed_test_mean)
            log(f'  -> seed={seed}: 5-fold mean test_f1={seed_test_mean:.4f}')

        mean_test = float(np.mean(seed_test_means))
        std_test = float(np.std(seed_test_means))
        duration = time.time() - combo_start

        log(f'>>>>> window={w}: test_f1 mean={mean_test:.4f} std={std_test:.4f} | duration={duration:.1f}s >>>>>')

        results.append((w, mean_test, std_test, duration))

    log('\n===== SUMMARY (sorted by mean test_f1, best first) =====')
    for w, mean_test, std_test, duration in sorted(results, key=lambda r: -r[1]):
        log(f'window={w:<4}  test_f1={mean_test:.4f}±{std_test:.4f}  duration={duration:.1f}s')

    log(f'\nFull log written to: {result_path}')
