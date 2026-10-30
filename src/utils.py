import torch
import numpy as np

import os
import datetime

import numpy as np
from numpy.random import randint

import pandas as pd

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import OneHotEncoder

from src import path
from src.models.model import HypergraphERCModel
from src.data.dataloader_iemocap import IEMOCAPDataset
from src.data.dataloader_meld import MELDRobertaDataset
from src.data.dataloader_cmu import CMUDataset

import sys
import copy

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_model_state(args, model_state, save_root, folder_index, folder_f1, folder_acc, folder_recon):
    """
    Save model state to local file.
    Automatically create the save directory if it doesn't exist.
    """
    mask_rate = args.mask_type.split('-')[-1]
    suffix_name = f'{args.dataset.lower()}_Graph_mask:{mask_rate}'
    mean_f1, mean_acc, mean_recon = np.mean(folder_f1), np.mean(folder_acc), np.mean(folder_recon)
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}_reconloss:{mean_recon:.4f}'
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"

    os.makedirs(save_root, exist_ok=True)
    
    model_path = os.path.join(save_root, f'{time_now}_{suffix_name}_classifier:{res_name}_{folder_index}.pt')
    torch.save(model_state, model_path)
    print(f'>>>>> Best model for folder {folder_index} saved to {model_path} >>>>>')
    return model_path

def save_teacher_checkpoint(model_state, dataset, fold_index):
    """
    Deterministic filename (no timestamp) so the student can find it by
    dataset + fold. Each fold's teacher must only have seen that fold's
    training sessions, or the student inherits a leak.
    """
    os.makedirs(path.TEACHER_DIR, exist_ok=True)
    ckpt_path = os.path.join(path.TEACHER_DIR, f'{dataset.lower()}_fold{fold_index}_teacher.pt')
    torch.save(model_state, ckpt_path)
    print(f'>>>>> Teacher checkpoint for fold {fold_index} saved to {ckpt_path} >>>>>')
    return ckpt_path


def load_teacher_checkpoint(dataset, fold_index, map_location=None):
    """
    Load the teacher checkpoint saved by save_teacher_checkpoint for this
    dataset + fold_index. Raises a clear error if train_teacher.py hasn't
    been run for that fold yet.
    """
    ckpt_path = os.path.join(path.TEACHER_DIR, f'{dataset.lower()}_fold{fold_index}_teacher.pt')
    assert os.path.exists(ckpt_path), (
        f'Teacher checkpoint not found: {ckpt_path}. '
        f'Run train_teacher.py for dataset={dataset} first.'
    )
    return torch.load(ckpt_path, map_location=map_location)


def save_student_checkpoint(model_state, dataset, fold_index):
    """
    Same convention as save_teacher_checkpoint, written to path.STUDENT_DIR
    instead — keeps student checkpoints from colliding with teacher ones
    for the same dataset/fold.
    """
    os.makedirs(path.STUDENT_DIR, exist_ok=True)
    ckpt_path = os.path.join(path.STUDENT_DIR, f'{dataset.lower()}_fold{fold_index}_student.pt')
    torch.save(model_state, ckpt_path)
    print(f'>>>>> Student checkpoint for fold {fold_index} saved to {ckpt_path} >>>>>')
    return ckpt_path


def update_best_model(val_fscore, best_f1, model, best_model_state):
    """
    Update the best model parameters based on the current val_fscore.
    """
    if val_fscore > best_f1:
        best_f1 = val_fscore
        best_model_state = copy.deepcopy(model.state_dict())
    return best_f1, best_model_state


def get_loaders(dataset, batch_size, num_workers, meld_path=None, audio_root=None, text_root=None, video_root=None, num_folder=5):

    if dataset == 'MELD':
        print("INFO: Loading MELD dataset...")
        if meld_path is None:
            raise ValueError("MELD dataset requires the 'meld_path' argument (path to .pkl file).")
        train_dataset = MELDRobertaDataset(meld_path, split='train')
        val_dataset = MELDRobertaDataset(meld_path, split='valid')
        test_dataset = MELDRobertaDataset(meld_path, split='test')

        # Create DataLoader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers
        )

        train_loaders = [train_loader]
        val_loaders = [val_loader]
        test_loaders = [test_loader]
        
        adim, tdim, vdim = train_dataset.get_featDim()

        return train_loaders, val_loaders, test_loaders, adim, tdim, vdim

    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        dataset = IEMOCAPDataset(
            label_path=path.PATH_TO_LABEL_Win[dataset],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root
        )

        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx:
                session_to_idx[session] = []
            session_to_idx[session].append(idx)

        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder):
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii:
                    train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]

            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(train_idxs),
                collate_fn=dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=False
            )
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(test_idxs),
                collate_fn=dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=False
            )

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        adim, tdim, vdim = dataset.get_featDim()

        return train_loaders, test_loaders, test_loaders, adim, tdim, vdim


def get_loaders_meld(features_path, batch_size, num_workers=0, roberta_idx=1):
    """
    MELD's official train/valid/test split. Returns single-element lists so
    callers can index [0] the same way they do for IEMOCAP's folds.
    """
    train_dataset = MELDRobertaDataset(features_path, split='train', roberta_idx=roberta_idx)
    val_dataset = MELDRobertaDataset(features_path, split='valid', roberta_idx=roberta_idx)
    test_dataset = MELDRobertaDataset(features_path, split='test', roberta_idx=roberta_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=train_dataset.collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=val_dataset.collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=test_dataset.collate_fn, num_workers=num_workers)

    adim, tdim, vdim = train_dataset.adim, train_dataset.tdim, train_dataset.vdim

    return [train_loader], [val_loader], [test_loader], adim, tdim, vdim


def get_loaders_cmu(dataset, batch_size, audio_root, text_root, video_root, num_workers=0):
    """
    CMU-MOSI / CMU-MOSEI official train/valid/test split. One CMUDataset holds
    every video; the three loaders are Subsets over the index ranges that
    CMUDataset.vids was built in (train, then valid, then test). Returns
    single-element lists so callers index [0] as they do for MELD.
    """
    ds = CMUDataset(
        label_path=path.PATH_TO_LABEL_Win[dataset],
        audio_root=audio_root,
        text_root=text_root,
        video_root=video_root,
    )

    n_tr, n_va = len(ds.trainVids), len(ds.valVids)
    idx_train = list(range(n_tr))
    idx_val = list(range(n_tr, n_tr + n_va))
    idx_test = list(range(n_tr + n_va, len(ds.vids)))
    assert [ds.vids[i] for i in idx_test] == ds.testVids, 'split index mismatch'

    def loader(idxs, shuffle):
        return DataLoader(
            Subset(ds, idxs), batch_size=batch_size, shuffle=shuffle,
            collate_fn=ds.collate_fn, num_workers=num_workers, pin_memory=False)

    print(f'{dataset}: {n_tr} train / {n_va} valid / {len(idx_test)} test videos')

    adim, tdim, vdim = ds.get_featDim()
    return ([loader(idx_train, True)], [loader(idx_val, False)],
            [loader(idx_test, False)], adim, tdim, vdim)


PROTO_BIN_CHOICES = (2, 3, 4, 5, 6, 7)
ROUND_BIN_CHOICES = (2, 3, 5, 7)
PROTO_BIN_MODES = ('uniform', 'round')


def bin_sentiment(scores, n_bins, mode='uniform'):
    """
    Map continuous sentiment in [-3, 3] onto n_bins pseudo-classes, so the
    prototype and rel distillation clusters have discrete classes to work
    with on the CMU datasets.

    `mode` picks how the range is cut:

      'uniform'  n_bins equal segments of width 6/n_bins, left-closed, so a
                 score sitting exactly on an edge goes to the upper bin.
      'round'    round(score) (or round(score*2/3) at 5, sign at 3), i.e. the
                 binning behind the Acc-7/Acc-5 numbers these datasets are
                 reported under. Only defined for ROUND_BIN_CHOICES, since it
                 always centres a bin on 0 and so cannot produce an even
                 count. Clipping at +-3 also leaves its two extreme bins half
                 the width of the interior ones, which starves them: on
                 MOSI's 1284 train utterances the smallest bin holds 24 at
                 n_bins=7 under 'round' but 67 under 'uniform', and even
                 'round' at 3 bins only reaches 53. 'uniform' is the default
                 because this binning is an internal distillation detail, not
                 a reported metric, so there is nothing to gain from
                 inheriting that convention's starved tails.

    Whether 0 is an edge or a bin centre follows from the parity of n_bins,
    and that decides what happens to the exact zeros:

      even n_bins  0 is an edge (2 -> [-3,0),[0,3); 6 -> integer edges), so no
                   bin can hold both signs. Left-closed bins would then sweep
                   every exact zero into the first positive bin -- 47% of it
                   on MOSEI at 6 bins -- so the zeros are dropped instead,
                   leaving each prototype to a single polarity. This is what
                   n_bins=2 has always done, and it costs no support: MOSEI's
                   smallest bin at 6 is 613 either way.
      odd n_bins   0 sits at the centre of the middle bin, which therefore
                   mixes negative and positive scores (6789 utterances on
                   MOSEI at 7 bins). Nothing is dropped, since zero belongs
                   in that bin as much as any neighbouring value.

    Each prototype is a mean over its bin, so a thin bin gives a noisy
    target -- check the per-bin counts the student prints before trusting a
    large n_bins on a small train split.

    Returns (bin_index [long], keep [bool]) over the same positions as
    `scores`.
    """
    if n_bins not in PROTO_BIN_CHOICES:
        raise ValueError(f'n_bins must be one of {PROTO_BIN_CHOICES}, got {n_bins}')
    if mode not in PROTO_BIN_MODES:
        raise ValueError(f'mode must be one of {PROTO_BIN_MODES}, got {mode!r}')
    if mode == 'round' and n_bins not in ROUND_BIN_CHOICES:
        raise ValueError(
            f"mode='round' centres a bin on 0 and so only supports "
            f'{ROUND_BIN_CHOICES}, got {n_bins}; use mode=\'uniform\' for an even count')

    if n_bins == 2:
        return (scores > 0).long(), scores != 0

    if mode == 'uniform':
        # Bin i is [-3 + 6i/n_bins, -3 + 6(i+1)/n_bins), left-closed, so a
        # score sitting exactly on an edge belongs to the upper bin. Getting
        # that right needs the tolerance: labels are stored as float32, and
        # e.g. 1.8 -- an exact edge at n_bins=5 -- is held as 1.7999999523,
        # just under it. Without the snap, 13.5% of MOSI's train split lands
        # in a bin decided by float rounding rather than by its value, and
        # the assignment flips with the dtype. 1e-6 is ~50000x smaller than
        # the smallest gap between distinct labels (0.05 on MOSI, 1/3 on
        # MOSEI), so it can never merge two genuinely different scores.
        u = (scores.double() + 3.0) * n_bins / 6.0
        idx = torch.clamp(torch.floor(u + 1e-6).long(), 0, n_bins - 1)
        if n_bins % 2 == 0:
            return idx, scores != 0
        return idx, torch.ones_like(scores, dtype=torch.bool)
    elif n_bins == 3:
        idx = torch.sign(scores).long() + 1
    elif n_bins == 5:
        idx = torch.clamp(torch.round(scores * 2.0 / 3.0), -2, 2).long() + 2
    else:
        idx = torch.clamp(torch.round(scores), -3, 3).long() + 3

    return idx, torch.ones_like(scores, dtype=torch.bool)


def regression_metrics(labels, preds, masks):
    """
    Scoring for the CMU sentiment datasets, following the convention their
    papers report (and the CPD-GNN reference implementation): utterances whose
    gold score is exactly 0 are dropped, and the sign of the remaining scores
    gives the binary negative/positive class.

    Dropping zeros is not cosmetic -- it removes 4.4% of MOSI and 21.9% of
    MOSEI -- so acc/F1 here are not comparable to numbers computed any other
    way. MAE and Pearson r are reported over every valid utterance, zeros
    included, since they need no binarisation.

    Returns (acc, f1, mae, corr).
    """
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    valid = np.asarray(masks).reshape(-1) == 1

    y, p = labels[valid], preds[valid]
    mae = float(np.mean(np.abs(y - p))) if y.size else float('nan')
    corr = float(np.corrcoef(y, p)[0, 1]) if y.size > 1 and p.std() > 0 else float('nan')

    nz = y != 0
    if nz.sum() == 0:
        return float('nan'), float('nan'), mae, corr
    y_bin, p_bin = (y[nz] > 0).astype(int), (p[nz] > 0).astype(int)
    acc = accuracy_score(y_bin, p_bin)
    f1 = f1_score(y_bin, p_bin, average='weighted')
    return acc, f1, mae, corr


def compute_class_weights(loader, n_classes, scheme='inverse'):
    """
    Class weights from a loader's own label distribution, counting only valid
    (unpadded) positions. Always pass the TRAIN loader -- deriving weights
    from valid/test would leak those splits into training.

    Normalized to mean 1 so switching schemes changes the balance between
    classes without changing the overall loss scale.

    The counting pass consumes the dataloader's shuffling RNG, so the RNG
    state is restored afterwards: enabling this must not, by itself, shift
    the training stream and make runs incomparable.
    """
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    counts = torch.zeros(n_classes)
    for data in loader:
        umask, label = data[7], data[8]
        valid = umask.reshape(-1).bool()
        counts += torch.bincount(label.reshape(-1)[valid].long(), minlength=n_classes).float()
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)

    freq = (counts / counts.sum()).clamp(min=1e-8)
    if scheme == 'inverse':
        w = 1.0 / freq
    elif scheme == 'inverse-sqrt':
        w = 1.0 / freq.sqrt()
    else:
        raise ValueError(f"scheme must be 'inverse' or 'inverse-sqrt', got {scheme!r}")
    return w / w.mean()


def set_seed(seed, cuda):
    """
    Seeding the RNGs alone is NOT enough for reproducibility: cuDNN would
    still pick whichever algorithm is fastest, which can differ between
    runs and drift the result. The two cudnn flags are what actually make
    same-seed runs bit-identical, at some cost in speed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(params, args):
    """
    Adam folds weight_decay into the gradient before the adaptive step,
    distorting it; AdamW decouples it. Since l2 is actively tuned here, the
    two optimizers are not interchangeable — lr/l2 tuned under one should be
    re-tuned for the other.
    """
    optimizer_name = getattr(args, 'optimizer', 'adam')
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[optimizer_name]
    return cls(params, lr=args.lr, weight_decay=args.l2)


def build_model(args, adim, tdim, vdim):
    model = HypergraphERCModel(
        adim, tdim, vdim,
        args.lstm_hidden_size,
        n_speakers=args.n_speakers,
        window_past=args.windowp,
        window_future=args.windowf,
        dataset=args.dataset,
        n_classes=args.n_classes,
        dropout=args.dropout,
        no_cuda=args.no_cuda,
        use_bilstm=args.use_bilstm,
        prompt_type=args.prompt_type,
        n_proto=args.n_proto,
        prompt_dim=args.prompt_dim,
        n_graph_heads=args.n_graph_heads,
        graph_layers=args.graph_layers,
        reliability_type=args.reliability_type,
        d_e=getattr(args, 'd_e', None),
        vue_reliability_type=args.vue_reliability_type,
        use_vue_reliability_gate=args.use_vue_reliability_gate,
        hg_layers=args.hg_layers,
    )

    print("\nFull Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")

    return model


# Construct model input features for multimodal fusion (host: main speaker, guest: other)
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2)
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1)
    tmask = tmask.unsqueeze(2).repeat(1, 1, featdim)
    select_feat = torch.where(tmask == 0, feat1, feat2)
    input_features.append(select_feat)
    return input_features


# Generate missing masks (mask) following CPM-Net method
def random_mask(view_num, input_len, missing_rate):
    """
    Randomly generate incomplete modality mask matrix to simulate multimodal missing scenarios.
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate

    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray()
        return view_preserve

    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num))
        return matrix

    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len

    error = 1
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()

        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)

        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))

        one_num_iter = one_num / (1 - a / one_num)

        ratio = one_num_iter / (view_num * alldata_len)

        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        matrix = ((matrix_iter + view_preserve) > 0).astype(int)

        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

    matrix = matrix[:input_len, :]
    return matrix

def get_class_names(dataset_name):
    """Return list of class names based on dataset name"""
    if dataset_name == 'IEMOCAPFour':
        return ['hap', 'sad', 'ang', 'neu']
    elif dataset_name == 'IEMOCAPSix':
        return ['hap', 'sad', 'ang', 'neu', 'exc', 'fru']
    elif dataset_name == 'MELD':
        return ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    else:
        return None


def save_results(args, folder_f1, folder_acc, all_labels, all_preds, save_path_base, all_names=None):
    """
    Save final classification metrics and confusion matrix to file.
    :param all_names: New parameter, list of names for all test samples
    """
    report_path = save_path_base + '_metrics.txt'
    print(f'Saving classification report to {report_path}')
    
    # Indentation level here is first level of function
    with open(report_path, 'w') as f:
        f.write("=======================================================\n")
        f.write(f"           Final Evaluation Results\n")
        f.write("=======================================================\n\n")
        
        mean_f1 = np.mean(folder_f1)
        std_f1 = np.std(folder_f1)
        mean_acc = np.mean(folder_acc)
        std_acc = np.std(folder_acc)
        
        f.write(f"Overall Performance ({args.num_folder}-fold cross-validation):\n")
        f.write(f"  - Average Weighted F1-Score: {mean_f1:.4f} (±{std_f1:.4f})\n")
        f.write(f"  - Average Accuracy:        {mean_acc:.4f} (±{std_acc:.4f})\n\n")
        
        f.write("-------------------------------------------------------\n")
        f.write("      Detailed Classification Report (Aggregated)\n")
        f.write("      (Based on test set predictions from the best\n")
        f.write("       epoch of each fold, determined by validation F1)\n")
        f.write("-------------------------------------------------------\n\n")

        class_names = get_class_names(args.dataset)
        
        # Generate detailed report and confusion matrix only if it is a classification task
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
            # Generate and write sklearn classification report
            report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
            f.write(report)
            
            # ----------------- Save confusion matrix as image -----------------
            # Note: plotting logic is suggested to be outside with open block, or ensure no interference with file writing
            # But for logical coherence, place it here first, as long as variable names don't conflict
            cm_path = save_path_base + '_confusion_matrix.png'
            print(f'Saving confusion matrix to {cm_path}')
            
            cm = confusion_matrix(all_labels, all_preds)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        annot_kws={"size": 16})  # Added this line, number 16 can be adjusted as needed
            
            plt.title(f'Confusion Matrix - {args.dataset}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close() # Close figure
        else:
            f.write("Regression task - Classification report and confusion matrix are not applicable.\n")

    if all_names is not None:
        print(f'Saving detailed predictions to {save_path_base}_predictions.xlsx')
        
        # Get class name mapping
        class_names = get_class_names(args.dataset)
        
        # Construct DataFrame data
        data_dict = {
            'Sample_Name': all_names,
            'True_Label_Index': all_labels.astype(int),
            'Predicted_Label_Index': all_preds.astype(int)
        }

        # If class names exist, try adding text label columns
        if class_names:
            try:
                # Handle classification tasks
                if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix', 'MELD']:
                    data_dict['True_Label'] = [class_names[i] for i in all_labels.astype(int)]
                    data_dict['Predicted_Label'] = [class_names[i] for i in all_preds.astype(int)]
            except IndexError:
                print("Warning: Label index out of range for class names, skipping text label columns.")

        # Create DataFrame
        df_pred = pd.DataFrame(data_dict)
        
        # Save as Excel (requires openpyxl: pip install openpyxl)
        try:
            df_pred.to_excel(save_path_base + '_predictions.xlsx', index=False)
        except Exception as e:
            print(f"Error saving Excel: {e}, falling back to CSV.")
            df_pred.to_csv(save_path_base + '_predictions.csv', index=False)
            
    print("Results summary and predictions saved successfully.")
