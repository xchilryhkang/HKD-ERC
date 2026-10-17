import os
import sys
import time
import argparse
import datetime
import copy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

# Make the repo root importable so `src` / `train` resolve from any cwd.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import path
from src.utils import Logger, get_loaders, build_model, random_mask, generate_inputs, save_teacher_checkpoint, make_optimizer, set_seed
from src.losses.loss import MaskedCELoss


def train_or_eval_model(args, model, cls_loss, dataloader, mask_rate, optimizer=None, train=False):
    preds, labels, masks = [], [], []
    losses = []

    cuda = torch.cuda.is_available() and not args.no_cuda
    model.train() if train else model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        speaker_info, umask, label = data[6], data[7], data[8]

        if args.dataset == 'MELD':
            batch_size, seq_len = speaker_info.shape[0], speaker_info.shape[1]
            qmask = torch.zeros(batch_size, seq_len, dtype=torch.long)
            qmask_9 = torch.argmax(speaker_info, dim=2).long()
        else:
            qmask = speaker_info

        adim, tdim, vdim = audio_host.size(2), text_host.size(2), visual_host.size(2)
        seqlen, batch = audio_host.size(0), audio_host.size(1)

        matrix = random_mask(3, seqlen * batch, mask_rate)
        audio_host_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
        text_host_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
        visual_host_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))
        matrix = random_mask(3, seqlen * batch, mask_rate)
        audio_guest_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1)))
        text_guest_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1)))
        visual_guest_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1)))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.cuda(), audio_host_mask.cuda()
            masked_text_host, text_host_mask = masked_text_host.cuda(), text_host_mask.cuda()
            masked_visual_host, visual_host_mask = masked_visual_host.cuda(), visual_host_mask.cuda()
            masked_audio_guest, audio_guest_mask = masked_audio_guest.cuda(), audio_guest_mask.cuda()
            masked_text_guest, text_guest_mask = masked_text_guest.cuda(), text_guest_mask.cuda()
            masked_visual_guest, visual_guest_mask = masked_visual_guest.cuda(), visual_guest_mask.cuda()
            qmask, umask, label = qmask.cuda(), umask.cuda(), label.cuda()
            if args.dataset == 'MELD':
                qmask_9 = qmask_9.cuda()

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        model_qmask = qmask_9 if args.dataset == 'MELD' else qmask

        masked_input_features = generate_inputs(
            masked_audio_host, masked_text_host, masked_visual_host,
            masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(
            audio_host_mask, text_host_mask, visual_host_mask,
            audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        log_prob, v_hat, gamma_vue, h_final, gammas_hg = model(
            masked_input_features, model_qmask, umask, lengths, input_features_mask,
            adim, tdim, vdim, label
        )

        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size(2))
        labels_ = label.view(-1)
        loss = cls_loss(lp_, labels_, umask)

        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    preds = np.argmax(preds, 1)

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_acc = accuracy_score(labels, preds, sample_weight=masks)
    avg_f1 = f1_score(labels, preds, sample_weight=masks, average='weighted')
    return avg_acc, avg_f1, avg_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MELD', help='IEMOCAPFour, IEMOCAPSix or MELD')
    parser.add_argument('--audio-feature', type=str, default='wav2vec-large-c-UTT')
    parser.add_argument('--text-feature', type=str, default='deberta-large-4-UTT')
    parser.add_argument('--video-feature', type=str, default='manet_UTT')
    parser.add_argument('--windowp', type=int, default=3)
    parser.add_argument('--windowf', type=int, default=3)
    parser.add_argument('--lstm_hidden_size', type=int, default=150)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_speakers', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-folder', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)
    parser.add_argument('--disable-bilstm', action='store_true', default=False)
    parser.add_argument('--n-graph-heads', type=int, default=4)
    parser.add_argument('--graph-layers', type=int, default=2)
    parser.add_argument('--reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--d-e', type=int, default=None,
                        help='VUE consensus dim; defaults to hidden_dim (= lstm_hidden_size * 2), '
                             'must match — see model.HypergraphERCModel.')
    parser.add_argument('--vue-reliability-type', type=str, default='feature', choices=['scalar', 'feature'])
    parser.add_argument('--disable-vue-reliability-gate', action='store_true', default=False)
    parser.add_argument('--hg-layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='See utils.make_optimizer for why adamw may be worth testing.')

    args = parser.parse_args()

    # Teacher always sees every modality: no missing-rate, no learned placeholder.
    args.mask_type = 'constant-0.0'
    args.prompt_type = 'none'
    args.n_proto = 10
    args.prompt_dim = None
    args.use_bilstm = not args.disable_bilstm
    args.use_vue_reliability_gate = not args.disable_vue_reliability_gate
    args.d_e = args.d_e  # kept as-is (None -> HypergraphERCModel resolves it)

    print(f"Setting random seed to: {args.seed}")
    cuda = torch.cuda.is_available() and not args.no_cuda
    set_seed(args.seed, cuda)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    save_log = os.path.join(path.LOG_DIR, 'teacher_hg', f'{args.dataset}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_now = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    sys.stdout = Logger(filename=f"{save_log}/{time_now}_teacher_hg_{args.dataset}.txt", stream=sys.stdout)

    if args.dataset == 'IEMOCAPFour':
        args.num_folder, args.n_classes, args.n_speakers = 5, 4, 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder, args.n_classes, args.n_speakers = 5, 6, 2
    elif args.dataset == 'MELD':
        args.num_folder, args.n_classes, args.n_speakers = 1, 7, 9

    print(args)

    print(f'\n========== Reading Data ==========\n')
    if args.dataset == 'MELD':
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=0, meld_path=path.PATH_MELD
        )
    else:
        audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
        text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
        video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
        train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders(
            audio_root=audio_root, text_root=text_root, video_root=video_root,
            num_folder=args.num_folder, batch_size=args.batch_size, dataset=args.dataset, num_workers=0
        )

    print(f'\n========== Training Teacher (full modality, per fold) ==========\n')

    fold_f1, fold_acc = [], []

    for ii in range(args.num_folder):
        print(f'>>>>> Teacher fold {ii+1}/{args.num_folder} >>>>>')
        train_loader = train_loaders[ii]
        val_loader = val_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        model = build_model(args, adim, tdim, vdim)
        cls_loss = MaskedCELoss()
        if cuda:
            model.cuda()
            cls_loss.cuda()

        optimizer = make_optimizer(model.parameters(), args)

        best_val_fscore = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            train_acc, train_fscore, train_loss = train_or_eval_model(
                args, model, cls_loss, train_loader, mask_rate=0.0, optimizer=optimizer, train=True)
            val_acc, val_fscore, val_loss = train_or_eval_model(
                args, model, cls_loss, val_loader, mask_rate=0.0, optimizer=None, train=False)

            print(f'epoch:{epoch+1}; train_fscore:{train_fscore:2.2%}; train_acc:{train_acc:2.2%}; train_loss:{train_loss}; '
                  f'val_fscore:{val_fscore:2.2%}; val_acc:{val_acc:2.2%}; val_loss:{val_loss}')

            if val_fscore > best_val_fscore + args.early_stop_delta:
                best_val_fscore = val_fscore
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

        print(f'>>>>> Teacher fold {ii+1}: best val_f1={best_val_fscore:.4f}, '
              f'test_f1={test_fscore:.4f}, test_acc={test_acc:.4f}, '
              f'duration={time.time()-start_time:.2f}s >>>>>')

        fold_f1.append(test_fscore)
        fold_acc.append(test_acc)

        save_teacher_checkpoint(best_model_state, args.dataset, fold_index=ii)

    print(f'\n====== Teacher summary: mean test F1={np.mean(fold_f1):.4f}, '
          f'mean test acc={np.mean(fold_acc):.4f} ======')
