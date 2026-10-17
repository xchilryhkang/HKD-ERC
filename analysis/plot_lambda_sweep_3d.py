"""
Plots the lambda_rel (lambda1) x lambda_prob (lambda2) sweep from
tune_lambda_prob_rel_iemocap.py as a 3D bar chart (WAF vs lambda1 x
lambda2), matching the visual convention of a reference figure (grouped
bars colored by lambda2, blue-to-green gradient, z-axis = WAF%).

Usage:
    python plot_lambda_sweep_3d.py --log output/tune/<sweep file>.txt \
        [--title "IEMOCAP (six-class)"] [--out output/tune/lambda_sweep_3d.svg]

Parses every ">>>>> lambda_rel=X lambda_prob=Y: sessionout_f1 mean=Z ..."
line in the log; if a (lambda1, lambda2) pair appears more than once (e.g.
a combo re-run after a resume), the LAST occurrence wins.
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


IEMOCAP_LINE_RE = re.compile(
    r'^>>>>> lambda_rel=([\d.]+) lambda_prob=([\d.]+): sessionout_f1 mean=([\d.]+)'
)
# MELD sweep reports both val_f1 (model-selection signal) and test_f1 (held-out,
# reported at the best-val checkpoint) -- test_f1 is the one comparable to
# IEMOCAP's sessionout_f1 for plotting (both are held-out generalization numbers).
MELD_LINE_RE = re.compile(
    r'^>>>>> lambda_rel=([\d.]+) lambda_prob=([\d.]+): val_f1 mean=([\d.]+) std=[\d.]+ \| '
    r'test_f1 mean=([\d.]+)'
)


def parse_log(log_path):
    results = {}  # (lambda1, lambda2) -> f1, last occurrence wins
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = MELD_LINE_RE.match(line)
            if m:
                l1, l2, test_f1 = float(m.group(1)), float(m.group(2)), float(m.group(4))
                results[(l1, l2)] = test_f1
                continue
            m = IEMOCAP_LINE_RE.match(line)
            if m:
                l1, l2, f1 = float(m.group(1)), float(m.group(2)), float(m.group(3))
                results[(l1, l2)] = f1
    return results


def plot_3d_bar(results, title, out_path):
    lambda1_vals = sorted({l1 for l1, _ in results})
    lambda2_vals = sorted({l2 for _, l2 in results})

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('winter')  # blue -> green, matches the reference figure
    # Wide along lambda2 (bars touch within a row -> solid "wall" per lambda1,
    # like the reference) and narrower along lambda1 (visible gap between rows).
    dy = 0.95
    dx = 0.55

    all_f1 = np.array(list(results.values())) * 100  # as WAF%
    z_floor = np.floor(all_f1.min() / 2) * 2 - 2  # a couple points below the min, rounded
    z_ceil = np.ceil(all_f1.max() / 2) * 2 + 2

    for j, l2 in enumerate(lambda2_vals):
        color = cmap(j / max(1, len(lambda2_vals) - 1))
        xs, ys, zs = [], [], []
        for i, l1 in enumerate(lambda1_vals):
            f1 = results.get((l1, l2))
            if f1 is None:
                continue
            xs.append(i)
            ys.append(j)
            zs.append(f1 * 100)
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        bottoms = np.full_like(zs, z_floor)
        heights = zs - z_floor
        ax.bar3d(xs - dx / 2, ys - dy / 2, bottoms, dx, dy, heights,
                  color=color, edgecolor='black', linewidth=0.6, shade=True)

    ax.set_xticks(range(len(lambda1_vals)))
    ax.set_xticklabels([f'{v:g}' for v in lambda1_vals])
    ax.set_yticks(range(len(lambda2_vals)))
    ax.set_yticklabels([f'{v:g}' for v in lambda2_vals])
    ax.set_xlabel(r'$\lambda_1$ (rel)', labelpad=10)
    ax.set_ylabel(r'$\lambda_2$ (prob)', labelpad=10)
    ax.set_zlabel('WAF', labelpad=8)
    ax.set_zlim(z_floor, z_ceil)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1, 1, 0.85))
    if title:
        ax.set_title(title, y=-0.1)

    fig.subplots_adjust(left=0.02, right=0.95, top=1.0, bottom=0.08)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'-> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--title', type=str, default='IEMOCAP (six-class)')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or args.log.replace('.txt', '_3d.svg')
    results = parse_log(args.log)
    print(f'Parsed {len(results)} unique (lambda1, lambda2) combos from {args.log}')
    plot_3d_bar(results, args.title, out_path)
