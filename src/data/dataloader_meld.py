import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd


class MELDRobertaDataset(Dataset):
    """
    Reads MELD from the single self-contained feature file (path.PATH_MELD,
    meld_emotion_semantic_features_roberta.pkl), whose 17 fields are:

        [0]      speakers, one-hot over up to 9 main characters
        [1], [2] labels (7-class emotion; the two are duplicates)
        [3]-[6]  RoBERTa-Large text variants, finetuned on emotion labels
        [7]-[10] RoBERTa-Large text variants, un-finetuned ("semantic")
        [11]     audio,  300d
        [12]     visual, 342d
        [13]     sentences
        [14]     trainVid (1038), [15] testVid (280), [16] validVid (114)

    The split is MELD's genuine official 3-way split, so 'test' is a real
    held-out set (an earlier loader had to fall back to the valid set as
    test because its source file carried no testVid -- that leak is gone
    along with the two-file merge this class used to do).

    Like the IEMOCAP loader, the only speaker track is treated as "host"
    and "guest" is filled with zeros -- MELD has no 2-party host/guest
    structure, speakers are one-hot over the main characters instead.

    roberta_idx selects which of the 4 EMOTION text variants to use (1-4 ->
    fields [3]-[6]); the semantic variants [7]-[10] are loaded into memory
    with the rest of the pickle but not used.
    """

    def __init__(self, features_path, split='train', roberta_idx=1):
        assert roberta_idx in (1, 2, 3, 4), f'roberta_idx must be 1-4, got {roberta_idx}'

        data = pickle.load(open(features_path, 'rb'), encoding='latin1')

        self.videoSpeakers = data[0]
        self.videoLabels = data[1]
        self.videoText = data[2 + roberta_idx]  # fields [3]-[6], 1-indexed via roberta_idx
        self.videoAudio = data[11]
        self.videoVisual = data[12]
        self.videoSentence = data[13]
        self.trainVid, self.testVid, self.validVid = data[14], data[15], data[16]

        if split == 'train':
            self.keys = sorted(list(self.trainVid))
        elif split == 'valid':
            self.keys = sorted(list(self.validVid))
        elif split == 'test':
            self.keys = sorted(list(self.testVid))
        else:
            raise ValueError(f"split must be 'train', 'valid' or 'test', got {split!r}")

        self.len = len(self.keys)
        if self.len == 0:
            print(f"WARNING: Loaded 0 samples for split '{split}'. Check your pkl file and split name.")
            self.adim, self.tdim, self.vdim = 0, 0, 0
        else:
            self.adim, self.tdim, self.vdim = self.get_featDim(split)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        vid = self.keys[index]
        audio_host = torch.FloatTensor(np.asarray(self.videoAudio[vid]))
        visual_host = torch.FloatTensor(np.asarray(self.videoVisual[vid]))
        text_host = torch.FloatTensor(np.stack(self.videoText[vid]))  # list of (1024,) -> [T, 1024]
        seq_len = audio_host.shape[0]
        audio_guest = torch.zeros(seq_len, self.adim)
        text_guest = torch.zeros(seq_len, self.tdim)
        visual_guest = torch.zeros(seq_len, self.vdim)
        speakers = torch.FloatTensor(np.asarray(self.videoSpeakers[vid]))
        mask = torch.FloatTensor([1] * seq_len)
        labels = torch.LongTensor(self.videoLabels[vid])
        return (audio_host, text_host, visual_host,
                audio_guest, text_guest, visual_guest,
                speakers, mask, labels, vid)

    def get_featDim(self, split_name=''):
        if self.len == 0:
            return 0, 0, 0
        example_vid = self.keys[0]
        adim = np.asarray(self.videoAudio[example_vid]).shape[1]
        tdim = len(self.videoText[example_vid][0])
        vdim = np.asarray(self.videoVisual[example_vid]).shape[1]
        print(f"INFO: For split '{split_name}', MELD feature dims => "
              f"audio: {adim}, text: {tdim}, video: {vdim}")
        return adim, tdim, vdim

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        datnew = []
        for i in range(len(dat.columns)):
            if i <= 5:
                datnew.append(pad_sequence(dat[i]))
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], batch_first=True))
            else:
                datnew.append(dat[i].tolist())
        return datnew
