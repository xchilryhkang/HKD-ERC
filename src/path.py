import os

# This file lives in src/, so the repo root (which holds dataset/ and
# output/) is one level up from it.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE, 'dataset')

DATA_DIR_Win = {
	'IEMOCAPFour': os.path.join(DATASET_ROOT, 'IEMOCAPFour'),
	'IEMOCAPSix': os.path.join(DATASET_ROOT, 'IEMOCAP'),
	'CMUMOSI': os.path.join(DATASET_ROOT, 'CMUMOSI'),
	'CMUMOSEI': os.path.join(DATASET_ROOT, 'CMUMOSEI'),
}

PATH_TO_FEATURES_Win = {
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'features'),
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'features'),
}

PATH_TO_LABEL_Win = {
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
}

# Self-contained: text, audio, visual, labels and the official split all
# live in this one file (see dataloader_meld.MELDRobertaDataset).
PATH_MELD = os.path.join(DATASET_ROOT, 'MELD', 'meld_emotion_semantic_features_roberta.pkl')

SAVED_ROOT = os.path.join(BASE, 'output')
RESULT_DIR = os.path.join(SAVED_ROOT, 'result')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
TEACHER_DIR = os.path.join(MODEL_DIR, 'teacher')
STUDENT_DIR = os.path.join(MODEL_DIR, 'student')
