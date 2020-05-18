import soundfile as sf
import os
import numpy as np
import argparse
import json
import yaml
from parsing.parse_chime6 import parse_chime6, get_session
from SSAD.utils.annotations import merge_intervals

parser = argparse.ArgumentParser("prep labels")
parser.add_argument("chime6_root", type=str)
parser.add_argument("diarization_file", type=str)
parser.add_argument("split", type=str)
parser.add_argument("conf", type=str)
parser.add_argument("out_folder", type=str)


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)

def _gen_frame_indices(
        data_length, size, step,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def preprocess_diarization(diarization, configs):
    # preprocess diarization phones with per speaker smoothing
    for sess in diarization.keys():
        for speaker in diarization[sess].keys():

            tmp = []
            for s, e in diarization[sess][speaker]:
                s -= configs["labels"]["collar_ph"] * fs  # tolerance over single phone
                e += configs["labels"]["collar_ph"] * fs
                tmp.append([s, e])
            diarization[sess][speaker] = merge_intervals(tmp, configs["labels"]["merge_ph"] * fs)  # merge phones if are x apart

    return diarization


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    with open(args.conf, "r") as f:
        configs = yaml.load(f)

    configs.update(args.__dict__)

    with open(args.diarization_file, "r") as f:
        diarization = json.load(f)

    fs = configs["data"]["fs"]

    diarization = preprocess_diarization(diarization, configs)

    meta = parse_chime6(args.chime6_root, args.split)

    # get labels for all sessions
    devices = {}
    for sess in diarization.keys():

        min_len = len(sf.SoundFile(meta[sess]["binaurals"][0]))
        # devices[sess]["devices"].extend(meta[sess]["binaurals"])

        for array in meta[sess]["arrays"].keys():
            #devices[sess]["devices"].extend(meta[sess]["arrays"][array])
            c_dev_len = len(sf.SoundFile(meta[sess]["arrays"][array][0]))  # all arrays are sample sync
            min_len = min(min_len, c_dev_len)

        dummy = np.zeros(int(np.ceil(min_len/(configs["feats"]["hop_size"]*fs))))

        for spk in diarization[sess].keys():
            if spk == "garbage":
                continue
            else:
                for s, e in diarization[sess][spk]:
                    s = int(np.floor(s / (configs["feats"]["hop_size"]*fs)))
                    e = int(np.floor(e / (configs["feats"]["hop_size"]*fs)))
                    dummy[s:e] += 1
        assert not np.where(dummy > 5)[0].any()
        sf.write(os.path.join(args.out_folder, "LABEL-{}.wav".format(sess)), dummy, fs, subtype="FLOAT")












