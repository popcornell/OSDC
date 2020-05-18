import argparse
import os
import json
import glob
from pathlib import Path
import re
import json
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score
import pandas as pd
import numpy as np
from scipy.signal import medfilt

parser = argparse.ArgumentParser("scoring")
parser.add_argument("preds_dir")
parser.add_argument("diarization_file")

def build_target_vector(sess_diarization, subsample=160):

    # get maxlen
    maxlen = max([sess_diarization[spk][-1][-1] for spk in sess_diarization.keys()])
    dummy = np.zeros(maxlen//subsample, dtype="uint8")

    for spk in diarization[sess].keys():
        if spk == "garbage":
            continue
        for s, e in diarization[sess][spk]:
            s = int(s/subsample)
            e = int(np.ceil(e/subsample))
            dummy[s:e] += 1
    return dummy

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def score(preds, target_vector):
    target_vector = np.clip(target_vector, 0, 2)
    target_vector_cat = target_vector
    target_vector = one_hot(target_vector, preds.shape[0])
    minlen = min(target_vector.shape[0], preds.shape[-1])
    target_vector = target_vector[:minlen, :]
    preds = preds[:, :minlen].T
    count_ap = average_precision_score(target_vector, preds, average=None)
    osd_ap = average_precision_score(target_vector_cat >= 2, np.sum(preds[:, 2:], -1))
    vad_ap = average_precision_score(target_vector_cat >= 1, np.sum(preds[:, 1:], -1))

    return count_ap, vad_ap, osd_ap


def process_preds(preds):

    # average all from same session
    # apply medfilter
    mat = []
    minlen = np.inf
    for i in preds:
        tmp = np.load(i)[0]
        minlen = min(minlen, tmp.shape[-1])
        mat.append(tmp)

    mat = [x[:, :minlen] for x in mat]
    mat = np.mean(np.stack(mat), 0)

    return mat #medfilt(mat, 5)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.diarization_file, "r") as f:
        diarization = json.load(f)

    preds_hash = {}
    preds = glob.glob(os.path.join(args.preds_dir, "*.npy"))

    for p in preds:
        session = re.findall("(S[0-9]+)", Path(p).stem)[0]
        if session not in preds_hash.keys():
            preds_hash[session] = [p]
        else:
            preds_hash[session].append(p)

    scores = {}
    for sess in diarization.keys():
        if sess not in preds_hash.keys():
            continue

        target_vector = build_target_vector(diarization[sess])
        preds = preds_hash[sess]
        preds = process_preds(preds)
        count_ap, vad_ap, osd_ap = score(preds, target_vector)
        scores[sess] = {"count_ap": count_ap, "vad_ap": vad_ap, "osd_ap": osd_ap }

    dt = pd.DataFrame.from_dict(scores)
    scores["TOTAL"] = {"count_ap": dt.iloc[0, :].mean(), "vad_ap": dt.iloc[1, :].mean(), "osd_ap": dt.iloc[2, :].mean()}
    dt = pd.DataFrame.from_dict(scores).to_json(os.path.join(args.preds_dir, "APs.json"))














