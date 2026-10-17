import os
import glob
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


## Get name2features mapping based on label file and feature path
def read_data(label_path, feature_root):

    # ============================ #
    #     Read information from label file
    # ============================ #
    names = []
    speakers = []

    # Load multiple variables from pkl file
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, testVid = pickle.load(
        open(label_path, "rb"), encoding='latin1')

    # All video IDs (containing both training and testing)
    vids = sorted(list(trainVid | testVid))

    # Iterate through each video to extract the name and speaker of each utterance
    for ii, vid in enumerate(vids):
        uids_video = videoIDs[vid]
        spks_video = videoSpeakers[vid]
        names.extend(uids_video)
        speakers.extend(spks_video)

    # ============================ #
    #      Load corresponding feature data
    # ============================ #
    features = []
    feature_dim = -1

    for ii, name in enumerate(names):
        speaker = speakers[ii]

        # Find the corresponding feature file or folder (supports prefix matching)
        feature_dir = glob.glob(os.path.join(feature_root, name + '*'))
        assert len(feature_dir) == 1, f"Found multiple or no feature files: {name}"
        feature_path = feature_dir[0]

        # Initialize the feature structure for this utterance, distinguishing between male and female (F/M)
        feature = {'F': [], 'M': []}

        if feature_path.endswith('.npy'):
            # Audio/Text features => Load npy file directly (one per utterance)
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze()
            feature[speaker].append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])
        else:
            # Video (e.g., FACET) features are in a folder containing multiple .npy files divided by face (e.g., F_001.npy)
            facenames = os.listdir(feature_path)
            for facename in sorted(facenames):
                assert 'F' in facename or 'M' in facename, f"face filename should contain F or M"
                facefeat = np.load(os.path.join(feature_path, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                if 'F' in facename:
                    feature['F'].append(facefeat)
                else:
                    feature['M'].append(facefeat)

        # Average multiple frames for each speaker modality to become a fixed-dimension vector
        for speaker in feature:
            single_feature = np.array(feature[speaker]).squeeze()
            if len(single_feature) == 0:
                single_feature = np.zeros((feature_dim, ))
            elif len(single_feature.shape) == 2:
                single_feature = np.mean(single_feature, axis=0)
            feature[speaker] = single_feature

        features.append(feature)

    # ============================ #
    #     Build name -> feature mapping
    # ============================ #
    print(f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'

    name2feats = {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]

    return name2feats, feature_dim




def _feature_cache_path(label_path, audio_root, text_root, video_root):
    """
    One pkl per (label file, audio/text/video feature choice) combo --
    changing --audio-feature/--text-feature/--video-feature naturally
    invalidates the cache since the path changes.
    """
    key = '__'.join([
        os.path.splitext(os.path.basename(label_path))[0],
        os.path.basename(audio_root),
        os.path.basename(text_root),
        os.path.basename(video_root),
    ])
    cache_dir = os.path.join(os.path.dirname(label_path), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{key}.pkl')


class IEMOCAPDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root):
        """
        Initialize dataset: Read features and labels, and build a dual-speaker feature alignment data structure.

        The expensive part (read_data's per-utterance glob+np.load loop over
        all three modalities, ~20k+ filesystem calls) is cached to a single
        pkl under <label_dir>/cache/ after the first load -- every later
        call with the same label file + feature choice just unpickles it
        instead of re-scanning the raw feature files.
        """
        cache_path = _feature_cache_path(label_path, audio_root, text_root, video_root)

        if os.path.exists(cache_path):
            print(f'Loading cached features from {cache_path} ...')
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            self.adim = cached['adim']
            self.tdim = cached['tdim']
            self.vdim = cached['vdim']
            self.max_len = cached['max_len']
            self.videoAudioHost = cached['videoAudioHost']
            self.videoTextHost = cached['videoTextHost']
            self.videoVisualHost = cached['videoVisualHost']
            self.videoAudioGuest = cached['videoAudioGuest']
            self.videoTextGuest = cached['videoTextGuest']
            self.videoVisualGuest = cached['videoVisualGuest']
            self.videoLabelsNew = cached['videoLabelsNew']
            self.videoSpeakersNew = cached['videoSpeakersNew']
            self.vids = cached['vids']

            (self.videoIDs, self.videoLabels, self.videoSpeakers, self.videoSentences,
             self.trainVid, self.testVid) = pickle.load(open(label_path, "rb"), encoding='latin1')

            print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
            return

        # ========================= #
        # Read feature data for three modalities
        # Returns mapping of name -> {'F': [...], 'M': [...]} and feature dimension
        # ========================= #
        name2audio, adim = read_data(label_path, audio_root)
        name2text, tdim = read_data(label_path, text_root)
        name2video, vdim = read_data(label_path, video_root)

        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim

        # ========================= #
        # Initialize dictionaries for multimodal, labels, speakers, etc.
        # ========================= #
        self.max_len = -1

        self.videoAudioHost = {}
        self.videoTextHost = {}
        self.videoVisualHost = {}

        self.videoAudioGuest = {}
        self.videoTextGuest = {}
        self.videoVisualGuest = {}

        self.videoLabelsNew = {}
        self.videoSpeakersNew = {}

        speakermap = {'F': 0, 'M': 1}

        # ========================= #
        # Load multiple structured information from label file
        # ========================= #
        (self.videoIDs,
         self.videoLabels,
         self.videoSpeakers,
         self.videoSentences,
         self.trainVid, self.testVid) = pickle.load(open(label_path, "rb"), encoding='latin1')

        self.vids = sorted(list(self.trainVid | self.testVid))

        for ii, vid in enumerate(self.vids):
            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]

            self.max_len = max(self.max_len, len(uids))

            # Initialize modality storage structure for current video
            self.videoAudioHost[vid] = []
            self.videoTextHost[vid] = []
            self.videoVisualHost[vid] = []
            self.videoAudioGuest[vid] = []
            self.videoTextGuest[vid] = []
            self.videoVisualGuest[vid] = []
            self.videoLabelsNew[vid] = []
            self.videoSpeakersNew[vid] = []

            for ii, uid in enumerate(uids):
                # Fill features for corresponding speakers into host (F) and guest (M)
                self.videoAudioHost[vid].append(name2audio[uid]['F'])
                self.videoTextHost[vid].append(name2text[uid]['F'])
                self.videoVisualHost[vid].append(name2video[uid]['F'])
                self.videoAudioGuest[vid].append(name2audio[uid]['M'])
                self.videoTextGuest[vid].append(name2text[uid]['M'])
                self.videoVisualGuest[vid].append(name2video[uid]['M'])
                self.videoLabelsNew[vid].append(labels[ii])
                self.videoSpeakersNew[vid].append(speakermap[speakers[ii]])

            # Convert to numpy array (easier for subsequent processing)
            self.videoAudioHost[vid] = np.array(self.videoAudioHost[vid])
            self.videoTextHost[vid] = np.array(self.videoTextHost[vid])
            self.videoVisualHost[vid] = np.array(self.videoVisualHost[vid])
            self.videoAudioGuest[vid] = np.array(self.videoAudioGuest[vid])
            self.videoTextGuest[vid] = np.array(self.videoTextGuest[vid])
            self.videoVisualGuest[vid] = np.array(self.videoVisualGuest[vid])
            self.videoLabelsNew[vid] = np.array(self.videoLabelsNew[vid])
            self.videoSpeakersNew[vid] = np.array(self.videoSpeakersNew[vid])

        with open(cache_path, 'wb') as f:
            pickle.dump(dict(
                adim=self.adim, tdim=self.tdim, vdim=self.vdim, max_len=self.max_len,
                videoAudioHost=self.videoAudioHost, videoTextHost=self.videoTextHost,
                videoVisualHost=self.videoVisualHost, videoAudioGuest=self.videoAudioGuest,
                videoTextGuest=self.videoTextGuest, videoVisualGuest=self.videoVisualGuest,
                videoLabelsNew=self.videoLabelsNew, videoSpeakersNew=self.videoSpeakersNew,
                vids=self.vids,
            ), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Cached processed features to {cache_path}')

    # ========================= #
    # Override __getitem__ method
    # Returns multimodal features and labels etc. for the video at the corresponding index
    # ========================= #
    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]),\
               torch.FloatTensor(self.videoTextHost[vid]),\
               torch.FloatTensor(self.videoVisualHost[vid]),\
               torch.FloatTensor(self.videoAudioGuest[vid]),\
               torch.FloatTensor(self.videoTextGuest[vid]),\
               torch.FloatTensor(self.videoVisualGuest[vid]),\
               torch.FloatTensor(self.videoSpeakersNew[vid]),\
               torch.FloatTensor([1]*len(self.videoLabelsNew[vid])),\
               torch.LongTensor(self.videoLabelsNew[vid]),\
               vid

    # Total length of dataset
    def __len__(self):
        return len(self.vids)

    # Return feature dimension for each modality
    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    # Return max sequence length (number of utterances)
    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    # Custom collate_fn, used for DataLoader to automatically pad sequences of different lengths
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
