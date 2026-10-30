# HypergraphERC: Hypergraph-based Robust Emotion Recognition in Conversation

A teacher-student framework for Emotion Recognition in Conversation (ERC) that stays
robust under **missing/incomplete modalities**. A full-modality **teacher** distills
into a missing-modality **student** via a 4-cluster knowledge-distillation loss, so the
student keeps performing well even when audio, text or video is partially unavailable
at inference time.

## Architecture

```
raw feature -> proj -> prompt -> BiLSTM -> ContextGraphStack (per modality: a / t / v)
                                                    |
                                                    v
                                VUEFusion (a, t, v -> v_hat, per utterance)
                                                    |
                                                    v
                HypergraphStack ({a, t, v, e} nodes, cross-modal message passing, L layers)
                                                    |
                                                    v
                      concat [h_a || h_t || h_v || h_e] -> Linear -> log_prob
```

- **ContextGraphStack** (`src/models/context_graph.py`): per-modality weighted GAT with
  speaker-consistency, multi-scale temporal-decay and content-reliability edge priors;
  residual-add (not concat) keeps every stage at one shared `hidden_dim`
  (`= lstm_hidden_size * 2`).
- **VUEFusion** (`src/models/vue.py`): fuses `a/t/v` into a consensus node `v_hat` per utterance.
- **HypergraphStack** (`src/models/hypergraph.py`): cross-modal message passing over
  `{a, t, v, e}` nodes.
- **prompt.py**: fills placeholder (missing) modality slots — `none` / `constant` /
  `prototype` prompting, used only by the student.

### Teacher / Student

- **Teacher**: full modality (`mask_rate=0`, `prompt_type=none`), trained with plain
  masked cross-entropy (`train_teacher*.py`).
- **Student**: missing modality (`mask_rate>0`), distilled from a frozen per-fold
  teacher checkpoint (`train_student*.py`). The total loss is 4 clusters:

  ```
  loss = CE(student, label)                                            [CE,    weight fixed = 1]
       + lambda_prob  * ( KD(student || teacher) + OT(student, teacher) )  [prob,  tunable]
       + lambda_proto * L_proto                                             [proto, tunable]
       + lambda_rel   * ( L_e_proto + L_node_rel )                          [rel,   tunable]
  ```

  `prob` matches response-level class probabilities (KD + Sinkhorn optimal transport);
  `proto` matches the student's representation to the teacher's class prototypes;
  `rel` matches the consensus node `e`'s own relational structure (to its class
  prototype, and to the other three nodes) between teacher and student feature spaces.
  See `src/losses/kd_losses.py` and the loss assembly in `train/train_student.py`.

## Datasets

Three dataset families, each with its own evaluation protocol (a dataset's
`train_*_<name>.py` script implements the right one for it):

- **IEMOCAP** (4-class or 6-class): 5-fold **leave-one-session-out cross-validation**.
  There is no separate validation session — early stopping tracks the held-out
  session's own F1 directly every epoch (standard ERC-paper practice), reported as
  `sessionout_f1`.
- **MELD** (7-class): a genuine train(1038)/valid(114)/test(280) split. Early stopping
  tracks the real valid F1; test F1 is reported at the best-valid checkpoint.
- **CMU-MOSI / CMU-MOSEI**: **sentiment regression**, not classification, on the
  official train/valid/test split (MOSI 52/10/31 videos = 1284/229/686 utterances;
  MOSEI 2249/300/676 = 16326/1871/4659). Three things differ from the ERC datasets,
  and all are handled in the data/train layer rather than the model:
  - Labels are continuous scores in [-3, 3], so the head emits **one scalar**
    (`n_classes=1`) trained with `MaskedMSELoss`. `model.py` skips `log_softmax` at
    that width, since over a single logit it is identically 0 and would kill the
    gradient.
  - Accuracy and weighted F1 follow the convention these datasets are reported
    under: utterances whose gold score is exactly **0 are dropped**, and the sign of
    the rest gives negative/positive. That removes 4.4% of MOSI and 21.9% of MOSEI,
    so the numbers are not comparable to any other binarisation. MAE and Pearson r
    are also reported, over every valid utterance including the zeros.
  - The distillation clusters that are per-class need discrete labels, so the score
    is binned (`--proto-bins`, see below); the prob cluster becomes student-vs-teacher
    MSE and Sinkhorn OT drops out, since neither KL nor OT has a class distribution
    to work with at a 1-wide head.
  - They are **mono-speaker** (every speaker field is the empty string), so
    `n_speakers=1`, the Guest half of each modality is zeros, and the speaker-aware
    edge weight in `ContextGraphStack` is constant 1 — `theta_beta` gets no
    gradient on these datasets.

## Project Structure

```
.
├── src/                            # the importable library — never run directly
│   ├── path.py                     # dataset/output path configuration
│   ├── utils.py                    # loaders, optimizer/seed helpers, checkpoint I/O
│   ├── models/
│   │   ├── model.py                # HypergraphERCModel (the architecture above)
│   │   ├── context_graph.py        # per-modality weighted-GAT message passing
│   │   ├── vue.py                  # VUEFusion (a/t/v -> consensus node v_hat)
│   │   ├── hypergraph.py           # cross-modal hypergraph message passing
│   │   ├── prompt.py               # missing-modality placeholder prompting
│   │   └── graph.py                # batch_graphify: utterances -> graph edges
│   ├── data/
│   │   ├── dataloader_iemocap.py   # IEMOCAP dataset + loaders
│   │   ├── dataloader_meld.py      # MELD dataset + loaders
│   │   └── dataloader_cmu.py       # CMU-MOSI/MOSEI dataset + loaders (regression)
│   └── losses/
│       ├── loss.py                 # MaskedCELoss, MaskedMSELoss
│       └── kd_losses.py            # KD/OT/prototype/node-relation distillation losses
│
├── train/                          # training entry points
│   ├── train_teacher.py            # shared teacher loop (train_or_eval_model)
│   ├── train_student.py            # shared student loop (train_or_eval_student)
│   ├── train_teacher_iemocap.py    # teacher, IEMOCAP 5-fold leave-one-session-out CV
│   ├── train_student_iemocap.py    # student, IEMOCAP 5-fold leave-one-session-out CV
│   ├── train_teacher_meld.py       # teacher, MELD genuine train/valid/test split
│   ├── train_student_meld.py       # student, MELD genuine train/valid/test split
│   ├── train_teacher_cmu.py        # teacher, CMU-MOSI/MOSEI sentiment regression
│   └── train_student_cmu.py        # student, CMU-MOSI/MOSEI sentiment regression
│
├── tune/                           # hyperparameter sweeps, one per (hyperparam, dataset)
│   ├── tune_lr_l2_iemocap.py       tune_lr_l2_meld.py
│   ├── tune_window_iemocap.py      tune_window_meld.py
│   ├── tune_lambda_proto_rel_iemocap.py   tune_lambda_proto_rel_meld.py
│   └── tune_dropout_meld.py        tune_heads_meld.py   tune_hidden_dim_meld.py
│
├── analysis/
│   ├── analyze_distribution.py     # t-SNE, confidence/drift, teacher-vs-student metrics
│   └── plot_lambda_sweep_3d.py     # 3D bar plot for a lambda_rel x lambda_prob sweep
│
├── scripts/
│   ├── prepare_iemocap.py          # pack IEMOCAP's loose .npy files into one pkl
│   ├── prepare_cmu.py              # same, for CMU-MOSI / CMU-MOSEI
│   ├── run_teacher_iemocap6.ps1 / .sh    run_student_iemocap6.ps1 / .sh
│   ├── run_teacher_meld.ps1              run_student_meld.ps1
│   └── run_pipeline_iemocap6.ps1   # full teacher -> student -> analysis pipeline
│
├── environment.yml                 # Conda environment (Python 3.8)
└── requirements.txt                # Pip-only install (e.g. for Kaggle/Colab notebooks)
```

Everything under `src/` is library code imported as `from src.utils import ...`;
everything under `train/`, `tune/` and `analysis/` is an entry point you run directly.
Entry points put the repo root on `sys.path` themselves, so they work from any working
directory without installing the package.

`dataset/`, `output/` and the local virtual environment folder are gitignored — see
**Data Preparation** and **Outputs** below.

## Environment Setup

### Option A — Conda

```bash
conda env create -f environment.yml
conda activate fsce_net

# PyTorch + PyG (pick the wheel matching your CUDA version)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.6.1
```

### Option B — Pip only (Kaggle / Colab / any notebook with a GPU preinstalled)

Kaggle/Colab notebooks already ship a GPU-matched `torch` build, so **don't**
reinstall `torch` from the command above there — just add the rest:

```bash
pip install -r requirements.txt
pip install torch-geometric   # torch-scatter/torch-sparse are optional for this codebase
```

Then verify your `torch.__version__` / CUDA version and, if `torch-geometric` fails to
import, install the matching `torch-scatter`/`torch-sparse` wheel from
https://data.pyg.org/whl/ for that exact torch+CUDA combo.

## Data Preparation

Download pre-processed features and lay them out to match `src/path.py`:

```
dataset/
├── IEMOCAP/                                   # 6-class (IEMOCAPSix)
│   ├── IEMOCAP_features_raw_6way.pkl          # labels, speakers, sentences
│   ├── features/                               # ~30k loose .npy files
│   │   ├── wav2vec-large-c-UTT/
│   │   ├── deberta-large-4-UTT/
│   │   └── manet_UTT/
│   └── cache/                                  # written on first run, see below
├── IEMOCAPFour/                                # 4-class (IEMOCAPFour)
│   ├── IEMOCAP_features_raw_4way.pkl
│   └── features/...
├── MELD/
│   └── meld_emotion_semantic_features_roberta.pkl   # everything: text + audio + visual + split
├── CMUMOSI/
│   ├── CMUMOSI_features_raw_2way.pkl           # labels, speakers, sentences, train/valid/test
│   └── features/...                            # same three extractors as IEMOCAP
└── CMUMOSEI/
    ├── CMUMOSEI_features_raw_2way.pkl
    └── features/...
```

- **IEMOCAP** pre-processed features: see the **SDR-GNN** / **GCNet** repositories.
- **MELD**: one self-contained file holding 8 RoBERTa-Large text variants (4
  emotion-finetuned, used by `--roberta-idx 1-4`; 4 un-finetuned "semantic", unused),
  300d audio, 342d visual, speakers, labels, and MELD's official
  train(1038)/valid(114)/test(280) split. Audio/visual originate from the **MMGCN**
  features.
- **CMU-MOSI / CMU-MOSEI**: the same `deberta-large-4-UTT` (1024d) /
  `wav2vec-large-c-UTT` (512d) / `manet_UTT` (1024d) extractors as IEMOCAP, so no
  dimension changes. The `_2way` in the pkl filename names the downstream task, not
  the stored label: the file holds the raw continuous score. Two layout differences
  from IEMOCAP, both handled by `dataloader_cmu.py`: `manet_UTT` is one flat `.npy`
  per utterance (monologue, one face) instead of a per-utterance folder, and ids are
  `<videoId>_<n>` with no zero padding, so features resolve by exact filename
  instead of IEMOCAP's prefix glob — which would match `_1` against `_10`..`_13`
  for 8.3% of MOSI and 5.0% of MOSEI ids.

If your paths differ, edit `src/path.py` directly (it resolves everything relative to
the repo root by default, so no absolute paths are hardcoded).

### Packing the loose .npy datasets into one file

IEMOCAP and the two CMU datasets ship as loose `.npy` files — ~30k, 6.6k and 68.6k
respectively — and scanning them is slow. Do it once, up front:

```bash
python scripts/prepare_iemocap.py --dataset IEMOCAPSix
python scripts/prepare_cmu.py --dataset CMUMOSI
python scripts/prepare_cmu.py --dataset CMUMOSEI
```

Each writes a single pkl under `dataset/<name>/cache/`, named after the label file and
the three feature choices:

```
IEMOCAP_features_raw_6way__wav2vec-large-c-UTT__deberta-large-4-UTT__manet_UTT.pkl
CMUMOSI_features_raw_2way__wav2vec-large-c-UTT__deberta-large-4-UTT__manet_UTT.pkl
```

The file is **self-contained**: once it exists, neither the label pkl nor the
`features/` tree is read again, so it is the only file for that dataset needed to
train. Training builds it on demand if it is missing, so this step is optional — it
just means the cost lands in an explicit command rather than in your first training
run. (MELD needs none of this: it already ships as one combined pkl.)

Because the name encodes the feature choices, a different `--audio-feature` /
`--text-feature` / `--video-feature` produces a different file instead of silently
reusing a mismatched one.

**Running elsewhere (Kaggle, Colab):** build it locally, then upload only that one
file rather than the loose tree — ~230MB for IEMOCAP, 68MB for CMUMOSI, 704MB for
CMUMOSEI. Keep it at `dataset/<name>/cache/<that exact name>` (point `DATASET_ROOT` in
`src/path.py` at wherever the data is mounted) and pass the same feature flags it was
built with. If anything is off, the loader says which files it wanted and could not
find.

## How to Run

### Teacher, then student (IEMOCAP, 6-class)

```bash
python train/train_teacher_iemocap.py --dataset IEMOCAPSix
python train/train_student_iemocap.py --dataset IEMOCAPSix --mask-type constant-0.1 --prompt-type none
```

### Teacher, then student (MELD)

```bash
python train/train_teacher_meld.py
python train/train_student_meld.py --mask-type constant-0.1 --prompt-type none
```

### Teacher, then student (CMU-MOSI / CMU-MOSEI, sentiment regression)

```bash
python train/train_teacher_cmu.py --dataset CMUMOSI
python train/train_student_cmu.py --dataset CMUMOSI --mask-type constant-0.1 --prompt-type none
```

`--lambda-prob` still scales the prob cluster, but on these datasets that cluster is
MSE between the student and teacher scalars: KL and Sinkhorn OT both need a
distribution over classes, which a 1-wide head cannot give, so OT drops out of the
sum. `--lambda-proto` / `--lambda-rel` work as before.

`--proto-bins {2,3,4,5,6,7}` chooses how many pseudo-classes the continuous score is
binned into for those two clusters, since prototypes are per-class
(`utils.bin_sentiment`), and `--proto-bin-mode {uniform,round}` chooses how the range
is cut.

- `uniform` (default) cuts [-3, 3] into equal segments of width `6/n_bins`,
  left-closed, so a score sitting exactly on an edge goes to the upper bin. At 2 bins
  that is `[-3,0)` and `(0,3]`; at 6, the edges are the integers.
- `round` reproduces the Acc-7/Acc-5 binning these datasets are reported under. It
  always centres a bin on 0, so it only accepts an odd count (2, 3, 5, 7) and raises
  otherwise. Clipping at +-3 also leaves its two extreme bins half the width of the
  interior ones, which starves them badly on a small train split.

**The parity of `--proto-bins` decides where 0 goes, and that is the substantive
choice:**

- **Even** — 0 is a bin *edge*, so no prototype can mix negative with positive
  scores. Left-closed bins would otherwise sweep every exact zero into the first
  positive bin (47% of it on MOSEI at 6 bins), so the zeros are dropped instead,
  keeping each prototype to a single polarity. That is what 2 bins has always done,
  and it costs no support: MOSEI's smallest bin at 6 is 613 either way. Since the
  reported metric *is* the sign, this is the better match.
- **Odd** — 0 sits at the *centre* of the middle bin, which therefore mixes both
  signs (6789 utterances on MOSEI at 7 bins). Nothing is dropped.

Each prototype is a mean over its bin, so the useful ceiling is set by the train
split, not by taste. Smallest bin in the train split, under `uniform`:

| bins | parity | MOSI (1284 utts) | MOSEI (16326 utts) |
| ---- | ------ | ---------------- | ------------------ |
| 2    | even   | 552              | 4738               |
| 3    | odd    | 303              | 2196               |
| 4    | even   | 218              | 1616               |
| 5    | odd    | 120              | 1018               |
| 6    | even   | 67               | 613                |
| 7    | odd    | 67               | 505                |

Under `round` the same counts collapse to 53 / 38 / 24 on MOSI at 3 / 5 / 7 bins. A
prototype is 1024-dimensional, so `round` at 7 bins would average 24 vectors in 1024
dimensions — worse than `uniform` at 7 (67) and worse even than `round` at 3 (53).
`uniform` is the default for that reason: this binning is an internal distillation
detail, never a reported number, so there is nothing to gain from inheriting the
Acc-7 convention's starved tails.

The defaults are **6 bins on CMUMOSEI and 4 on CMUMOSI** — both even, so every
prototype is one polarity, and both well supported (613 and 218; MOSI at 6 would drop
to 67). The student prints the per-bin support every run plus a warning when the
smallest bin falls under 100.

`lr`/`l2` are carried over from MELD and have **not** been tuned on CMU yet.

Key student flags: `--mask-type constant-<rate>` (missing-modality rate at train+eval
time), `--prompt-type {none,constant,prototype}`, `--lambda-prob/--lambda-proto/
--lambda-rel` (distillation loss weights, each default 1.0; CE is always fixed at 1).
Run any script with `--help` for the full flag list.

On Windows, `scripts/run_teacher_iemocap6.ps1` / `scripts/run_student_iemocap6.ps1`
(PowerShell) and their `.sh` equivalents (Git Bash/WSL) wrap the same scripts with the
project's current tuned defaults, and can be run from anywhere.

## Hyperparameter Tuning

Each script in `tune/` sweeps one hyperparameter for one dataset, tracking the same
held-out metric as the corresponding `train/` script. Results go to `output/tune/*.txt`.
Long sweeps support `--max-combos` / `--skip-combos` / `--append-to` so they can be split
across sessions. Example:

```bash
python tune/tune_lambda_proto_rel_iemocap.py --mask-rate 0.3
python analysis/plot_lambda_sweep_3d.py --log output/tune/<sweep file>.txt
```

## Analysis

`analysis/analyze_distribution.py` loads a trained teacher+student pair and reports, per fold:
teacher-vs-student accuracy/F1, prediction agreement, Jensen-Shannon divergence between
their class-probability distributions, prototype-structure correlation, and t-SNE plots
(concatenated and per-modality, both class-colored and source-colored overlays).

```bash
python analysis/analyze_distribution.py --dataset IEMOCAPSix --folds 0,1,2,3,4 --mask-rate 0.1
```

## Outputs

Everything written at runtime goes under `output/` (gitignored): `output/model/{teacher,
student}/` (checkpoints), `output/log/` (training logs), `output/tune/` (sweep results +
plots), `output/analysis/` (t-SNE/drift plots + metrics).

## Acknowledgements

IEMOCAP/MELD pre-processed features build on prior work from the **GCNet**,
**SDR-GNN** and **MMGCN** repositories.
