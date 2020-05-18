import os
import json
from glob import glob


def get_speaker(file):
    assert is_array(file) == False

    return file.split("/")[-1].split("_")[1].split(".wav")[0]

def get_session(file):
    return file.split("/")[-1].split("_")[0]


def get_device(file):
    return file.split("/")[-1].split("_")[1].split(".")[0]


def get_channel(file):
    return file.split("/")[-1].split("_")[1].split(".")[1]


def is_array(file):

    return file.split("/")[-1].split("_")[1].split(".")[0][0] == "U"

def parse_chime6(chime6_root, split, custom_json=None, excluded=["S03", "S19", "S24"]):

    wavs_list = glob(os.path.join(chime6_root, "audio", split, "*wav"))
    if custom_json:
        transcriptions = glob(os.path.join(chime6_root, custom_json, split, "*json"))
    else:
        transcriptions = glob(os.path.join(chime6_root, "transcriptions", split, "*json"))

    sessions = list(set([get_session(x) for x in wavs_list]))
    sessions = sorted(sessions, key=lambda x: int(x.strip("S")))

    sess_hash = {}

    for sess in sessions:
        print("Parsing session {}".format(sess))
        if sess in excluded:
            continue
        for i in transcriptions:

            if i.split("/")[-1].split(".json")[0] == sess:
                with open(i, "r") as f:
                    annotation = json.load(f)

                break
        assert sess == annotation[0]["session_id"]

        sess_hash[sess] = {"arrays": {}, "binaurals": [], "transcriptions": annotation}

        for wav in wavs_list:

            if sess != get_session(wav):
                continue

            # is an array
            if is_array(wav):
                name = get_device(wav)

                if name not in sess_hash[sess]["arrays"].keys():
                    sess_hash[sess]["arrays"][name] = [wav]
                else:
                    sess_hash[sess]["arrays"][name].append(wav)
            else:
                # is binaural
                sess_hash[sess]["binaurals"].append(wav)

    # parse array channels

    for sess, elem in sess_hash.items():

        for arr in elem["arrays"].keys():
            elem["arrays"][arr] = sorted(elem["arrays"][arr], key=lambda x: int(get_channel(x).strip("CH")))

        if sess not in ["S01", "S21"]:
            assert len(sess_hash[sess]["binaurals"]) == 4

    return sess_hash