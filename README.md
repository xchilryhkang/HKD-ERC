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

- **ContextGraphStack** (`context_graph.py`): per-modality weighted GAT with
  speaker-consistency, multi-scale temporal-decay and content-reliability edge priors;
  residual-add (not concat) keeps every stage at one shared `hidden_dim`
  (`= lstm_hidden_size * 2`).
- **VUEFusion** (`vue.py`): fuses `a/t/v` into a consensus node `v_hat` per utterance.
- **HypergraphStack** (`hypergraph.py`): cross-modal message passing over
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
  See `kd_losses.py` and `train_student.py`'s module docstring for the full derivation.

## Datasets

Two datasets, two different evaluation protocols (a dataset's `train_*_<name>.py`
script implements the right one for it):

- **IEMOCAP** (4-class or 6-class): 5-fold **leave-one-session-out cross-validation**.
  There is no separate validation session — early stopping tracks the held-out
  session's own F1 directly every epoch (standard ERC-paper practice), reported as
  `sessionout_f1`.
- **MELD** (7-class): a genuine train(1038)/valid(114)/test(280) split. Early stopping
  tracks the real valid F1; test F1 is reported at the best-valid checkpoint.

## Project Structure

```
.
├── erc/                            # the importable library — never run directly
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
│   │   └── dataloader_meld.py      # MELD dataset + loaders (merges two feature files)
│   └── losses/
│       ├── loss.py                 # MaskedCELoss
│       └── kd_losses.py            # KD/OT/prototype/node-relation distillation losses
│
├── train/                          # training entry points
│   ├── train_teacher.py            # shared teacher loop (train_or_eval_model)
│   ├── train_student.py            # shared student loop (train_or_eval_student)
│   ├── train_teacher_iemocap.py    # teacher, IEMOCAP 5-fold leave-one-session-out CV
│   ├── train_student_iemocap.py    # student, IEMOCAP 5-fold leave-one-session-out CV
│   ├── train_teacher_meld.py       # teacher, MELD genuine train/valid/test split
│   └── train_student_meld.py       # student, MELD genuine train/valid/test split
│
├── tune/                           # hyperparameter sweeps, one per (hyperparam, dataset)
│   ├── tune_lr_l2_iemocap.py       tune_lr_l2_meld.py
│   ├── tune_window_iemocap.py      tune_window_meld.py
│   ├── tune_lambda_prob_rel_iemocap.py   tune_lambda_prob_rel_meld.py
│   └── tune_dropout_meld.py        tune_heads_meld.py   tune_hidden_dim_meld.py
│
├── analysis/
│   ├── analyze_distribution.py     # t-SNE, confidence/drift, teacher-vs-student metrics
│   └── plot_lambda_sweep_3d.py     # 3D bar plot for a lambda_rel x lambda_prob sweep
│
├── scripts/                        # shell wrappers with the project's tuned defaults
│   ├── run_teacher_iemocap6.ps1 / .sh    run_student_iemocap6.ps1 / .sh
│   ├── run_teacher_meld.ps1              run_student_meld.ps1
│   └── run_pipeline_iemocap6.ps1   # full teacher -> student -> analysis pipeline
│
├── environment.yml                 # Conda environment (Python 3.8)
└── requirements.txt                # Pip-only install (e.g. for Kaggle/Colab notebooks)
```

Everything under `erc/` is library code imported as `from erc.utils import ...`;
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

Download pre-processed features and lay them out to match `path.py`:

```
dataset/
├── IEMOCAP/                                   # 6-class (IEMOCAPSix)
│   ├── IEMOCAP_features_raw_6way.pkl
│   └── features/
│       ├── wav2vec-large-c-UTT/
│       ├── deberta-large-4-UTT/
│       └── manet_UTT/
├── IEMOCAPFour/                                # 4-class (IEMOCAPFour)
│   ├── IEMOCAP_features_raw_4way.pkl
│   └── features/...
└── MELD/
    ├── MELD_features_raw1.pkl                  # audio(300d)/visual(342d) features
    └── meld_emotion_features_roberta.pkl       # text (4 RoBERTa variants) + labels + split
```

- **IEMOCAP** pre-processed features: see the **SDR-GNN** / **GCNet** repositories.
- **MELD** audio/visual features: see the **MMGCN** repository
  (`MELD_features_raw1.pkl`). Text features come from a separate RoBERTa-Large
  extraction (`meld_emotion_features_roberta.pkl`, emotion-finetuned, genuine
  train/valid/test split) — `dataloader_meld.py`'s `MELDRobertaDataset` merges the two
  by dialogue key.

If your paths differ, edit `path.py` directly (it resolves everything relative to the
repo root by default, so no absolute paths are hardcoded).

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
python tune/tune_lambda_prob_rel_iemocap.py --mask-rate 0.3
python analysis/plot_lambda_sweep_3d.py --log output/tune/<sweep file>.txt
```

## Analysis

`analyze_distribution.py` loads a trained teacher+student pair and reports, per fold:
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
