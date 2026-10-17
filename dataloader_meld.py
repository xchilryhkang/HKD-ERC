import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd

class MyMELDDataset(Dataset):
    """
    A Dataset class specifically for reading MELD dataset .pkl files and adapting to your existing model input format.
    It treats the current speaker features as "Host" and fills "Guest" features with zero vectors.
    (Final corrected version: automatically uses the validation set if the test set does not exist)
    """
    def __init__(self, path, split='train'):
        """
        Initialization function to load pre-processed MELD pkl files.

        :param path: Path to the MELD_features.pkl file.
        :param split: String specifying which dataset split to load ('train', 'valid', or 'test').
        """
        loaded_data = pickle.load(open(path, 'rb'))
        
        # Safer assignment
        self.videoIDs = loaded_data[0]
        self.videoSpeakers = loaded_data[1]
        self.videoLabels = loaded_data[2]
        self.videoText = loaded_data[3]
        self.videoAudio = loaded_data[4]
        self.videoVisual = loaded_data[5]
        self.videoSentence = loaded_data[6]
        
        self.trainVid = loaded_data[7] if len(loaded_data) > 7 else None
        self.validVid = loaded_data[8] if len(loaded_data) > 8 else None
        self.testVid = loaded_data[9] if len(loaded_data) > 9 else None
        
        if split == 'train':
            if self.trainVid is None: raise ValueError("Train IDs ('trainVid') not found or is None in pkl file.")
            self.keys = sorted(list(self.trainVid))
        
        elif split == 'valid':
            if self.validVid is None: raise ValueError("Validation IDs ('validVid') not found or is None in pkl file.")
            self.keys = sorted(list(self.validVid))

        elif split == 'test':
            # Check if test set IDs exist
            if self.testVid is not None:
                print("INFO: Found and using the 'testVid' for the test set.")
                self.keys = sorted(list(self.testVid))
            # If test set IDs are None, use validation set IDs as a substitute
            elif self.validVid is not None:
                print("WARNING: 'testVid' not found or is None. Using 'validVid' for the test set as requested.")
                self.keys = sorted(list(self.validVid))
            # If both are None, raise an error
            else:
                raise ValueError("Both 'testVid' and 'validVid' are not found or are None. Cannot create test set.")
        else:
            raise ValueError(f"split arugment must be one of 'train', 'valid', or 'test'. Got '{split}' instead.")
        
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
        audio_host = torch.FloatTensor(self.videoAudio[vid])
        text_host = torch.FloatTensor(self.videoText[vid])
        visual_host = torch.FloatTensor(self.videoVisual[vid])
        seq_len = audio_host.shape[0]
        audio_guest = torch.zeros(seq_len, self.adim)
        text_guest = torch.zeros(seq_len, self.tdim)
        visual_guest = torch.zeros(seq_len, self.vdim)
        speakers = torch.FloatTensor(self.videoSpeakers[vid])
        mask = torch.FloatTensor([1] * seq_len)
        labels = torch.LongTensor(self.videoLabels[vid])
        return (audio_host, text_host, visual_host,
                audio_guest, text_guest, visual_guest,
                speakers, mask, labels, vid)

    def get_featDim(self, split_name=''):
        if self.len == 0: return 0, 0, 0
        example_vid = self.keys[0]
        adim = self.videoAudio[example_vid].shape[1]
        tdim = self.videoText[example_vid].shape[1]
        vdim = self.videoVisual[example_vid].shape[1]
        print(f"INFO: For split '{split_name}', MELD feature dims => audio: {adim}, text: {tdim}, video: {vdim}")
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


class MELDRobertaDataset(Dataset):
    """
    Merges two MELD pkl files by dialogue key (verified index-aligned via
    exact sentence match, and the audio/visual arrays are numerically
    identical between files that carry them -- see path.py's comments):

      - roberta_path (default: meld_emotion_features_roberta.pkl): lean,
        emotion-only file -- speakers (one-hot), labels (7-class emotion),
        4 RoBERTa-Large text variants (1024d each, finetuned on emotion
        labels -- see the ECERC paper's Preprocessing section), sentences,
        and the genuine train(1038)/valid(114)/test(280) split matching
        MELD's official sizes. No audio/visual, no un-finetuned "semantic"
        roberta variants -- kept out deliberately to avoid loading ~280MB
        of unused text features.
      - audio_visual_path (default: MELD_features_raw1.pkl): real 300d
        audio / 342d visual features (this file's own split is NOT used --
        see MyMELDDataset's fallback-to-valid-as-test behavior for why).

    Like MyMELDDataset, treats the only speaker track as "host" and fills
    "guest" with zeros (MELD doesn't have IEMOCAP's 2-party host/guest
    structure; speakers are one-hot over up to 9 main characters instead).

    roberta_idx selects which of the 4 text variants to use (1-4, maps to
    the roberta pkl's fields [3]-[6]); defaults to 1.
    """

    def __init__(self, roberta_path, audio_visual_path, split='train', roberta_idx=1):
        assert roberta_idx in (1, 2, 3, 4), f'roberta_idx must be 1-4, got {roberta_idx}'

        data = pickle.load(open(roberta_path, 'rb'), encoding='latin1')
        av_data = pickle.load(open(audio_visual_path, 'rb'), encoding='latin1')

        self.videoSpeakers = data[0]
        self.videoLabels = data[1]
        self.videoText = data[2 + roberta_idx]  # fields [3]-[6], 1-indexed via roberta_idx
        self.videoAudio = av_data[4]
        self.videoVisual = av_data[5]
        self.videoSentence = data[7]
        self.trainVid, self.testVid, self.validVid = data[8], data[9], data[10]

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
            print(f"WARNING: Loaded 0 samples for split '{split}'. Check your pkl files and split name.")
            self.adim, self.tdim, self.vdim = 0, 0, 0
        else:
            self.adim, self.tdim, self.vdim = self.get_featDim(split)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        vid = self.keys[index]
        audio_host = torch.FloatTensor(self.videoAudio[vid])
        visual_host = torch.FloatTensor(self.videoVisual[vid])
        text_host = torch.FloatTensor(np.stack(self.videoText[vid]))  # list of (1024,) -> [T, 1024]
        seq_len = audio_host.shape[0]
        audio_guest = torch.zeros(seq_len, self.adim)
        text_guest = torch.zeros(seq_len, self.tdim)
        visual_guest = torch.zeros(seq_len, self.vdim)
        speakers = torch.FloatTensor(self.videoSpeakers[vid])
        mask = torch.FloatTensor([1] * seq_len)
        labels = torch.LongTensor(self.videoLabels[vid])
        return (audio_host, text_host, visual_host,
                audio_guest, text_guest, visual_guest,
                speakers, mask, labels, vid)

    def get_featDim(self, split_name=''):
        if self.len == 0: return 0, 0, 0
        example_vid = self.keys[0]
        adim = self.videoAudio[example_vid].shape[1]
        tdim = len(self.videoText[example_vid][0])
        vdim = self.videoVisual[example_vid].shape[1]
        print(f"INFO: For split '{split_name}', MELD (roberta) feature dims => "
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
