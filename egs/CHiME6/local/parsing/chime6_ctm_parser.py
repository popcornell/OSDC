import re
import os
from .parse_chime6 import parse_chime6
from SSAD.utils.annotations import time_to_samples

class Utterance(object):

    __slots__ = ["session", "phones", "start", "stop"]

    def __init__(self, session, phones, start, stop):

        self.session = session
        self.phones = phones # list ph phones objects
        self.start = start
        self.stop = stop

    def __len__(self):
        return self.stop - self.start

class Phone(object):

    __slots__ = ["session", "speaker", "start_utt", "stop_utt", "start_ph", "stop_ph", "word", "confidence"]

    def __init__(self, session, speaker, start_utt, stop_utt, start_ph, stop_ph, word, confidence):
        self.session = session
        self.speaker = speaker
        self.start_utt = start_utt
        self.stop_utt = stop_utt
        self.start_ph = start_ph
        self.stop_ph = stop_ph
        self.word = word
        self.confidence = confidence


def parse_ctm(ctm_file, channel="L", fs=16000):

    phones = []

    with open(ctm_file, "r") as f:

        for line in f:
            if line.startswith("rev"):
                continue
            assert len(line.split(" ")) >= 8
            tmp = line.split(" ")
            utt_id, ch_num, start_phone, end_phone, word, confidence = tmp[:6]

            ch = utt_id.split(".")[1][0]
            #assert ch in ["L" , "R"]
            if ch != channel:
                continue

            speaker = re.findall("(P[0-9]+)", utt_id)
            assert len(speaker) == 1
            speaker = speaker[0]

            session = re.findall("(S[0-9]+)", utt_id)
            assert len(session) == 1
            session = session[0]

            utt_boundaries = re.findall("([0-9]+-[0-9]+)", utt_id)[0]
            assert len(utt_boundaries.split("-")) == 2

            start_utt, end_utt = [int(x)*fs//100 for x in utt_boundaries.split("-")]
            start_phone = int(float(start_phone)*fs)
            end_phone = int(float(end_phone)*fs + start_phone)

            if start_phone == end_phone:
                print("Garbage line {}".format(line))
                continue

            c_phone = Phone(session, speaker, start_utt, end_utt, start_phone, end_phone, word, confidence=1.0)
            phones.append(c_phone)

    return phones



def get_utterances_phones(chime6_root, split, ctm_file, chime6_sess_json_folder=None,
                    excluded=("S03", "S19", "S24")):
    '''

    :param chime6_root:
    :param split:
    :param ctm_file:
    :param chime6_sess_json_folder:
    :param excluded:
    :param garbage_words:
    :return:
    '''

    if not chime6_sess_json_folder:
        chime6_sess_json_folder = os.path.join(chime6_root, "transcriptions/")

    meta = parse_chime6(chime6_root, split, chime6_sess_json_folder)
    # here we have trasncriptions plus files for chime6

    examples = []

    tot_missing = 0
    tot_discarded = 0
    tot_samples = 0
    tot_samples_missing = 0
    tot_samples_discarded = 0

    tot_phones = parse_ctm(ctm_file)

    for sess in meta.keys():

        sess_missing = 0
        sess_discarded = 0
        sess_samples = 0
        sess_samples_missing = 0
        sess_samples_discarded = 0

        if sess in excluded:
            print("Excluding Session {} because of recording issues see CHiME-5 website".format(sess))
            continue

        print("Parsing examples for Session {}".format(sess))

        utterances = meta[sess]["transcriptions"]

        phones = [x for x in tot_phones if x.session == sess]

        for utt in utterances:
            start = time_to_samples(utt["start_time"])
            stop = time_to_samples(utt["end_time"])
            words = utt["words"]
            # find phones in each utterance

            c_len = stop - start
            sess_samples += c_len

            c_phones = []
            found = False
            for phone in phones:

                if ((start - 160) <= phone.start_utt <= (start + 160)) and (
                        (stop - 160) <= phone.stop_utt <= (stop + 160)) and (phone.speaker == utt["speaker"]):
                    c_phones.append(phone)
                    found = True

            if found == False:
                print("Missing alignment for utterance {}".format(utt))
                sess_samples_missing += stop - start
                sess_missing += 1
                continue

            # if len([x for x in c_phones if x.word not in ["sil", "<eps>", "spn", "eps"]]) >= len(words.split(" ")): #if less some word its not inside
            examples.append(Utterance(sess, c_phones, start, stop))
            # else:
            #   print("Discarding utterance because some words are missing form its alignment {}".format(utt))
            #  sess_discarded += 1
            # sess_samples_discarded += stop - start

        print("Session {} total samples: {} samples".format(sess, sess_samples))
        print("Missing from .ctm N {} {} samples".format(sess_missing, sess_samples_missing))
        print("Discarded N {} {} samples".format(sess_discarded, sess_samples_discarded))
        print("Total Speech Time: {} s".format(sess_samples / 16000))

        tot_samples_missing += sess_samples_missing
        tot_samples += sess_samples
        tot_missing += sess_missing
        tot_discarded += sess_discarded
        tot_samples_discarded += sess_samples_discarded

    # print examples statistic:
    print("Total Examples: N {} {} samples".format(len(examples), tot_samples))
    print("Missing from .ctm N {} {} samples".format(tot_missing, tot_samples_missing))
    print("Discarded N {} {} samples".format(tot_discarded, tot_samples_discarded))
    print("Total Speech Time: {} s".format(tot_samples / 16000))

    return examples




if __name__ == "__main__":
    ph = parse_ctm("/media/sam/cb915f0e-e440-414c-bb74-df66b311d09d/backup_after_stage14/ctm_edits.modified")
    print(len(ph))