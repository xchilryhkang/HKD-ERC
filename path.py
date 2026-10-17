import os

BASE = os.path.dirname(os.path.abspath(__file__))
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

PATH_MELD = os.path.join(DATASET_ROOT, 'MELD', 'MELD_features_raw1.pkl')

# MELD_features_raw1.pkl has real audio(300d)/visual(342d) features but no
# genuine 3-way split (testVid is None -- see dataloader_meld.py's
# MyMELDDataset fallback-to-valid-as-test behavior). Still used as the
# audio/visual source below, since its own (lack of a) split isn't what
# matters for that role.
#
# meld_emotion_features_roberta.pkl is the LEAN emotion-only text file:
# speakers/labels/4 RoBERTa-Large text variants (finetuned on emotion
# labels)/sentences, plus a genuine train(1038)/valid(114)/test(280) split
# matching MELD's official sizes -- no audio/visual, no un-finetuned
# "semantic" roberta variants (those live in the larger sibling file,
# meld_emotion_semantic_features_roberta.pkl, deliberately not used here
# to avoid loading ~280MB of unused text features).
#
# dataloader_meld.MELDRobertaDataset merges PATH_MELD_ROBERTA (text/
# labels/split) with PATH_MELD_AUDIOVISUAL (audio/visual) by dialogue key
# -- verified index-aligned via exact sentence match, and the audio/visual
# arrays are numerically identical between the two "roberta" files and
# MELD_features_raw1.pkl, so pairing with the lean one is safe.
PATH_MELD_AUDIOVISUAL = PATH_MELD
PATH_MELD_ROBERTA = os.path.join(DATASET_ROOT, 'MELD', 'meld_emotion_features_roberta.pkl')

# Paths for model saving, data caching, and logging
SAVED_ROOT = os.path.join(BASE, 'output')
RESULT_DIR = os.path.join(SAVED_ROOT, 'result')
LOG_DIR = os.path.join(SAVED_ROOT, 'log') 
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
TEACHER_DIR = os.path.join(MODEL_DIR, 'teacher')  # per-fold full-modality teacher checkpoints for KD
STUDENT_DIR = os.path.join(MODEL_DIR, 'student')  # per-fold incomplete-modality student checkpoints
