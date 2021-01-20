import soundfile as sf
import os
import numpy as np
import argparse
import json
import yaml
from osdc.utils.annotations import merge_intervals
import glob
from pathlib import Path

parser = argparse.ArgumentParser("prep labels")
parser.add_argument("ami_audio_root", type=str)
parser.add_argument("diarization_file", type=str)
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
    audio_files = glob.glob(os.path.join(args.ami_audio_root, "**/*.wav"), recursive=True)

    devices = {}
    for f in audio_files:
        sess = Path(f).stem.split(".")[0]
        if sess not in devices.keys():
            devices[sess] = [f]
        else:
            devices[sess].append(f)
    #meta = parse_chime6(args.chime6_root, args.split)

    # get labels for all sessions
    for sess in diarization.keys():
        if sess not in devices.keys():
            print("Skipping SESSION !!")
            continue
        min_len = min([len(sf.SoundFile(x)) for x in devices[sess]])
        dummy = np.zeros(int(np.ceil(min_len/(configs["feats"]["hop_size"]*fs))))

        for spk in diarization[sess].keys():
            if spk == "garbage":
                continue
            else:
                for s, e in diarization[sess][spk]:
                    s = int(np.floor(s / (configs["feats"]["hop_size"]*fs)))
                    e = int(np.floor(e / (configs["feats"]["hop_size"]*fs)))
                    dummy[s:e] += 1
        #assert not np.where(dummy > 4)[0].any()
        dummy = np.clip(dummy, 0, 4)
        sf.write(os.path.join(args.out_folder, "LABEL-{}.wav".format(sess)), dummy, fs, subtype="FLOAT")












