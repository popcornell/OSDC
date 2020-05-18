import os
import json
from .parse_chime6 import parse_chime6
import numpy as np
from .chime6_ctm_parser import get_utterances_phones


def get_diarization(chime6_root, split, ctm_file, out_dir, chime6_sess_json_folder=None,
                    excluded=("S03", "S19", "S24"), garbage_words=("[laughs]", "[inaudible]", "[laughts]")):
    '''
    :param chime6_root:
    :param split:
    :param ctm_file:
    :param chime6_sess_json_folder:
    :param excluded:
    :param garbage_words:
    :return:
    '''

    utterances = get_utterances_phones(chime6_root, split, ctm_file, chime6_sess_json_folder, excluded)
    if not chime6_sess_json_folder:
        chime6_sess_json_folder = os.path.join(chime6_root, "transcriptions/")
    meta = parse_chime6(chime6_root, split, chime6_sess_json_folder)
    # get diarization from utterances with phone activity

    diarization = {}

    for sess in meta.keys():
        if sess in excluded:
            # EXCLUDE BAD SESSIONS AGAIN useless to parse them!!
            continue

        speakers_dic = {'garbage': []}

        for binaurals in meta[sess]["binaurals"]:
            speaker_name = binaurals.split("/")[-1].split("_")[1].split(".wav")[0]
            speakers_dic[speaker_name] = []

        diarization[sess] = speakers_dic

    # when garbage the speaker will be garbage IN PLACE CHANGE

    if garbage_words:
        for ex in utterances:
            for ph in ex.phones:
                if ph.word in garbage_words:
                    ph.speaker = 'garbage'


    # initialized dict parsing examples
    for sess in diarization.keys():

        for speaker in diarization[sess].keys():
            found = False
            for ex in utterances:
                if ex.session != sess:
                    continue
                for ph in ex.phones:
                    if ph.speaker == speaker:  # match
                        found = True
                        abs_start = ph.start_utt + ph.start_ph
                        abs_stop = ph.stop_ph + ph.start_utt
                        diarization[sess][speaker].append([abs_start, abs_stop])
            if found == False:
                print("CRITICAL !! No utterance for this speaker something is wrong ! {}".format(speaker))

    # sorting
    for sess in diarization.keys():
        for speaker in diarization[sess].keys():
            diarization[sess][speaker] = sorted(diarization[sess][speaker], key=lambda x: x[0])

    ### compute stats ###

    stats = [0] * 5
    garbage = 0
    ov_garbage = 0

    for sess in diarization.keys():
        max_len = -1
        for speaker in diarization[sess].keys():
            max_len = max(diarization[sess][speaker][-1][-1], max_len)

        dummy_wav = np.zeros(max_len, dtype='uint8')
        garbage_wav = np.zeros(max_len, dtype='uint8')

        for speaker in diarization[sess].keys():
            for seg in diarization[sess][speaker]:
                if speaker != 'garbage':
                    dummy_wav[seg[0]: seg[1]] += 1
                else:
                    garbage_wav[seg[0]: seg[1]] = 1

        for i in range(len(stats)):
            stats[i] += len(np.where(dummy_wav == i)[0])

        garbage += np.sum(garbage_wav)
        ov_garbage += np.sum(np.logical_and(garbage_wav, dummy_wav))

    print("Total Overlapped Speech Time: {} s".format(np.sum([x for x in stats[2:]]) / 16000))
    print("Total Silence Speaker Time: {} s".format(stats[0] / 16000))
    print("Total Single Speaker Time: {} s".format(stats[1] / 16000))
    print("Total 2 Speaker Time: {} s".format(stats[2] / 16000))
    print("Total 3 Speaker Time: {} s".format(stats[3] / 16000))
    print("Total 4 Speaker Time: {} s".format(stats[4] / 16000))
    print("Total Garbage Speaker Time: {} s".format(garbage / 16000))
    print("Total Garbage Overlap Speaker Time: {} s".format(ov_garbage / 16000))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, split) + ".json", "w") as f:
        json.dump(diarization, f, indent=4)


if __name__ == "__main__":



    get_diarization("/media/sam/cb915f0e-e440-414c-bb74-df66b311d09d/CHiME6/", "train",
                    "/media/sam/cb915f0e-e440-414c-bb74-df66b311d09d/backup_after_stage14/ctm_edits.segmented", "/tmp", chime6_sess_json_folder=None,
                    excluded=("S03", "S19", "S24"), garbage_words=("[laughs]", "[inaudible]", "[laughts]"))

