import argparse
import json
import numpy as np
import os

parser = argparse.ArgumentParser("get diar stats")
parser.add_argument("diar_file")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.diar_file, "r") as f:
        diarization = json.load(f)

    stats = [0]*5
    for sess in diarization.keys():
        maxlen = max([diarization[sess][spk][-1][-1] for spk in diarization[sess].keys()])

        dummy = np.zeros(int(np.ceil(maxlen/160)), dtype="uint8")

        for spk in diarization[sess].keys():
            if spk == "garbage":
                continue
            for s, e in diarization[sess][spk]:
                s = int(s/160)
                e = int(e/160)
                dummy[s:e] += 1

        for i in range(len(stats)):
            stats[i] += len(np.where(dummy == i)[0])
        assert not np.where(dummy > 5)[0].any()

    print("TOT SILENCE {}".format(stats[0]))
    print("TOT 1 SPK {}".format(stats[1]))
    print("TOT 2 SPK {}".format(stats[2]))
    print("TOT 3 SPK {}".format(stats[3]))
    print("TOT 4 SPK {}".format(stats[4]))






