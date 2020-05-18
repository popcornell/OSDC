import argparse
import re
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser("get diarization from rttm")
parser.add_argument("rttm_file")
parser.add_argument("out_file")
parser.add_argument("--fs", type=int , default=16000)

if __name__ == "__main__":



    args = parser.parse_args()
    os.makedirs(Path(args.out_file).parent, exist_ok=True)

    diarization = {}
    with open(args.rttm_file, "r") as f:
        for line in f:
            session = re.findall("(S[0-9]+)", line)[0]
            start = int(float(line.split(" ")[3])*args.fs)
            stop= start + int(float(line.split(" ")[4])*args.fs)
            speaker = line.split(" ")[7]
            assert speaker.startswith("P")
            if session not in diarization.keys():
                diarization[session] = {speaker: [[start, stop]]}
            else:
                if speaker not in diarization[session].keys():
                    diarization[session][speaker] = [[start, stop]]
                else:
                    diarization[session][speaker].append([start, stop])


    with open(args.out_file, "w") as f:
        json.dump(diarization, f, indent=4)
