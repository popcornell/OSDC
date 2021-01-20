import argparse
import os
import glob
import json
from pathlib import Path
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser("Parsing AMI .xml annotation "
                                 "to .json file which contains diarization for each speaker")
parser.add_argument("xml_dir")
parser.add_argument("falign_dir")
parser.add_argument("output_json")

def parse_falign_file(falign_file):

    diarization = {}
    for i, spk in enumerate(falign_file):
        tree = ET.parse(spk)
        diarization["spk{}".format(i)] = []
        for seg in tree.getroot().getchildren():
            start = int(float(seg.attrib["transcriber_start"])*16000)
            end = int(float(seg.attrib["transcriber_end"])*16000)
            diarization["spk{}".format(i)].append([start, end])
        if len(diarization["spk{}".format(i)]) ==0:
            del diarization["spk{}".format(i)]

    return diarization

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(Path(args.output_json).parent, exist_ok=True)
    xml_sessions = glob.glob(os.path.join(args.xml_dir, "*.xml"))
    assert len(xml_sessions) > 0, "empty folder ?"

    falign_sessions = glob.glob(os.path.join(args.falign_dir, "*.txt"))
    falign_sessions = set([Path(f).stem.split("_")[-1] for f in falign_sessions])


    sessions = {}
    for f in xml_sessions:
        sess = Path(f).stem.split(".")[0]
        if sess not in falign_sessions:
            continue
        if sess not in sessions.keys():
            sessions[sess] = [f]
        else:
            sessions[sess].append(f)

    diarization = {}
    for sess in sessions.keys():
        diarization[sess] = parse_falign_file(sessions[sess])

    with open(args.output_json, "w") as f:
        json.dump(diarization, f, indent=4)