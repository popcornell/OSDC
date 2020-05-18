import argparse
import re
import json
import os
from pathlib import Path
import glob
from SSAD.utils.annotations import time_to_samples

parser = argparse.ArgumentParser("get diarization from rttm")
parser.add_argument("json_dir")
parser.add_argument("out_file")
parser.add_argument("--fs", type=int , default=16000)

if __name__ == "__main__":

    args = parser.parse_args()
    os.makedirs(Path(args.out_file).parent, exist_ok=True)

    jsons = glob.glob(os.path.join(args.json_dir, "*.json"))


    diarization = {}

    for j in jsons:
        with open(j, "r") as f:
            transc = json.load(f)

        # get speaker list
        speakers = {}
        sess = transc[0]["session_id"]
        if sess in ["S03", "S19", "S24"]:
            continue
        for utt in transc:

            start = time_to_samples(utt["start_time"])
            end = time_to_samples(utt["end_time"])
            spk = utt["speaker"]

            if spk not in speakers.keys():
                speakers[spk] = [[start, end]]
            else:
                speakers[spk].append([start, end])

        diarization[sess] = speakers

    with open(args.out_file, "w") as f:
        json.dump(diarization, f, indent=4)
