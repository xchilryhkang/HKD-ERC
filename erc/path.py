import os

# This file lives in erc/, so the repo root (which holds dataset/ and
# output/) is one level up from it.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE, 'dataset')

DATA_DIR_Win = {
	'IEMOCAPFour': os.path.join(DATASET_ROOT, 'IEMOCAPFour'), # IEMOCAP 4-class directory
	'IEMOCAPSix': os.path.join(DATASET_ROOT, 'IEMOCAP'),      # IEMOCAP 6-class directory
}

PATH_TO_FEATURES_Win = {
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'features'),
}

PATH_TO_LABEL_Win = {
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),# IEMOCAP 6-class labels
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),# IEMOCAP 4-class labels
}

# The single self-contained MELD feature file: speakers, labels, 8
# RoBERTa-Large text variants (fields [3]-[6] finetuned on emotion labels,
# [7]-[10] un-finetuned "semantic"), audio (300d), visual (342d),
# sentences, and the genuine train(1038)/valid(114)/test(280) split
# matching MELD's official sizes.
#
# This replaces an earlier two-file setup (a lean emotion-only text file
# merged with MELD_features_raw1.pkl for audio/visual). Its audio/visual
# arrays were verified numerically identical to MELD_features_raw1.pkl's,
# and its text/labels/split identical to the lean file's, so reading this
# one file changes nothing but the plumbing. It costs ~180MB more RAM
# (the 4 semantic text variants are loaded but unused by default).
PATH_MELD = os.path.join(DATASET_ROOT, 'MELD', 'meld_emotion_semantic_features_roberta.pkl')

# Paths for model saving, data caching, and logging
SAVED_ROOT = os.path.join(BASE, 'output')
RESULT_DIR = os.path.join(SAVED_ROOT, 'result')
LOG_DIR = os.path.join(SAVED_ROOT, 'log') 
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
TEACHER_DIR = os.path.join(MODEL_DIR, 'teacher')  # per-fold full-modality teacher checkpoints for KD
STUDENT_DIR = os.path.join(MODEL_DIR, 'student')  # per-fold incomplete-modality student checkpoints
