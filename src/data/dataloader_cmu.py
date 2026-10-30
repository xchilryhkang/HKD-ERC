import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def read_data(label_path, feature_root):
    """
    name -> feature vector, for one modality.

    Unlike IEMOCAP's reader this resolves the file by exact name rather than
    a prefix glob: CMU utterance ids are '<videoId>_<n>' with no zero padding,
    so '..._1' prefix-matches '..._10'..'..._13' (8.3% of MOSI ids, 5.0% of
    MOSEI's). It also keeps a single vector per utterance instead of an
    F/M pair -- CMU is monologue, see CMUDataset.
    """
    videoIDs, _, _, _, trainVids, valVids, testVids = pickle.load(
        open(label_path, 'rb'), encoding='latin1')

    names = []
    for vid in sorted(videoIDs):
        names.extend(videoIDs[vid])

    features = []
    feature_dim = -1

    for name in names:
        feature_path = os.path.join(feature_root, name + '.npy')
        feature_dir = os.path.join(feature_root, name)

        frames = []
        if os.path.exists(feature_path):
            single = np.load(feature_path).squeeze()
            feature_dim = max(feature_dim, single.shape[-1])
            frames.append(single)
        elif os.path.isdir(feature_dir):
            # Same fallback IEMOCAP needs: a folder of per-face .npy files.
            for facename in sorted(os.listdir(feature_dir)):
                facefeat = np.load(os.path.join(feature_dir, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                frames.append(facefeat)
        else:
            raise FileNotFoundError(f'No feature for {name!r} under {feature_root}')

        single = np.array(frames).squeeze()
        if len(single) == 0:
            single = np.zeros((feature_dim, ))
        elif len(single.shape) == 2:
            single = np.mean(single, axis=0)
        features.append(single)

    print(f'Input feature {os.path.basename(feature_root)} ===> dim is {feature_dim}; '
          f'No. sample is {len(names)}')
    assert len(names) == len(features), 'Error: len(names) != len(features)'

    return {n: f for n, f in zip(names, features)}, feature_dim


def _feature_cache_path(label_path, audio_root, text_root, video_root):
    key = '__'.join([
        os.path.splitext(os.path.basename(label_path))[0],
        os.path.basename(audio_root),
        os.path.basename(text_root),
        os.path.basename(video_root),
    ])
    cache_dir = os.path.join(os.path.dirname(label_path), 'cache')
    return os.path.join(cache_dir, f'{key}.pkl')


class CMUDataset(Dataset):
    """
    CMU-MOSI / CMU-MOSEI, laid out so the IEMOCAP/MELD training loop can
    consume it unchanged.

    Two things differ from IEMOCAP and both are handled here rather than in
    the model:

    * Mono-speaker. Every utterance's speaker field is the empty string, so
      speakermap is {'': 0}, n_speakers is 1, and the Guest half of each
      modality is all zeros. generate_inputs() selects Host everywhere, so
      nothing downstream sees the zeros; the speaker-aware edges in
      ContextGraphStack collapse to a single edge type, which is correct for
      a monologue.
    * Labels are continuous sentiment in [-3, 3], not class indices. They are
      returned as-is; binarising for metrics (drop 0, sign gives the class)
      happens at scoring time -- see utils.regression_metrics.

    The pkl carries a genuine train/valid/test split (3 fields, not IEMOCAP's
    2), exposed as self.trainVids / valVids / testVids. Feature reading is
    cached to one self-contained pkl under <label_dir>/cache/, same as
    IEMOCAPDataset, so a notebook only needs that one file.
    """

    def __init__(self, label_path, audio_root, text_root, video_root):
        cache_path = _feature_cache_path(label_path, audio_root, text_root, video_root)

        if os.path.exists(cache_path):
            print(f'Loading cached features from {cache_path} ...')
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            for k, v in cached.items():
                setattr(self, k, v)
            print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; '
                  f'video dimension: {self.vdim}')
            return

        missing = [r for r in (label_path, audio_root, text_root, video_root)
                   if not os.path.exists(r)]
        if missing:
            raise FileNotFoundError(
                f'No feature cache at {cache_path}, and these are needed to build one '
                f'but do not exist: {missing}'
            )

        name2audio, adim = read_data(label_path, audio_root)
        name2text, tdim = read_data(label_path, text_root)
        name2video, vdim = read_data(label_path, video_root)

        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.max_len = -1

        self.videoAudioHost, self.videoTextHost, self.videoVisualHost = {}, {}, {}
        self.videoAudioGuest, self.videoTextGuest, self.videoVisualGuest = {}, {}, {}
        self.videoLabelsNew, self.videoSpeakersNew = {}, {}

        (videoIDs, videoLabels, videoSpeakers, videoSentences,
         trainVids, valVids, testVids) = pickle.load(open(label_path, 'rb'), encoding='latin1')

        self.trainVids = sorted(trainVids)
        self.valVids = sorted(valVids)
        self.testVids = sorted(testVids)
        # Order matters: get_loaders_cmu slices index ranges off this list.
        self.vids = self.trainVids + self.valVids + self.testVids

        speakermap = {'': 0}

        for vid in self.vids:
            uids = videoIDs[vid]
            labels = videoLabels[vid]
            speakers = videoSpeakers[vid]

            self.max_len = max(self.max_len, len(uids))

            self.videoAudioHost[vid] = np.array([name2audio[u] for u in uids])
            self.videoTextHost[vid] = np.array([name2text[u] for u in uids])
            self.videoVisualHost[vid] = np.array([name2video[u] for u in uids])
            self.videoAudioGuest[vid] = np.zeros((len(uids), adim))
            self.videoTextGuest[vid] = np.zeros((len(uids), tdim))
            self.videoVisualGuest[vid] = np.zeros((len(uids), vdim))
            self.videoLabelsNew[vid] = np.array(labels, dtype=np.float32)
            self.videoSpeakersNew[vid] = np.array([speakermap[s] for s in speakers])

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'adim': self.adim, 'tdim': self.tdim, 'vdim': self.vdim,
                'max_len': self.max_len,
                'videoAudioHost': self.videoAudioHost,
                'videoTextHost': self.videoTextHost,
                'videoVisualHost': self.videoVisualHost,
                'videoAudioGuest': self.videoAudioGuest,
                'videoTextGuest': self.videoTextGuest,
                'videoVisualGuest': self.videoVisualGuest,
                'videoLabelsNew': self.videoLabelsNew,
                'videoSpeakersNew': self.videoSpeakersNew,
                'vids': self.vids,
                'trainVids': self.trainVids,
                'valVids': self.valVids,
                'testVids': self.testVids,
            }, f)
        print(f'Cached features to {cache_path}')

    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]),\
               torch.FloatTensor(self.videoTextHost[vid]),\
               torch.FloatTensor(self.videoVisualHost[vid]),\
               torch.FloatTensor(self.videoAudioGuest[vid]),\
               torch.FloatTensor(self.videoTextGuest[vid]),\
               torch.FloatTensor(self.videoVisualGuest[vid]),\
               torch.FloatTensor(self.videoSpeakersNew[vid]),\
               torch.FloatTensor([1] * len(self.videoLabelsNew[vid])),\
               torch.FloatTensor(self.videoLabelsNew[vid]),\
               vid

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; '
              f'video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        datnew = []
        dat = pd.DataFrame(data)
        for i in dat:
            if i <= 5:
                datnew.append(pad_sequence(dat[i]))
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], True))
            else:
                datnew.append(dat[i].tolist())
        return datnew
