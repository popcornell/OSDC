from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from osdc.utils.oladd import _gen_frame_indices
import random
from pysndfx import AudioEffectsChain

class OnlineFeats(Dataset):

    def __init__(self, ami_audio_root, label_root, configs, segment=300, probs=None, synth=None):

        self.configs = configs
        self.segment = segment
        self.probs = probs
        self.synth = None

        audio_files = glob.glob(os.path.join(ami_audio_root, "**/*.wav"), recursive=True)
        for f in audio_files:
            if len(sf.SoundFile(f)) < self.segment:
                print("Dropping file {}".format(f))
        labels = glob.glob(os.path.join(label_root, "*.wav"))
        lab_hash = {}

        for l in labels:
            l_sess = str(Path(l).stem).split("-")[-1]
            lab_hash[l_sess] = l

        devices_hash = {}
        devices = []
        for f in audio_files:
            sess = Path(f).stem.split(".")[0]
            if sess not in lab_hash.keys():
                print("Skip session because we have no labels for it")
                continue
            devices.append(f)
            if sess not in devices_hash.keys():
                devices_hash[sess] = [f]
            else:
                devices_hash[sess].append(f)

        self.devices = devices
        self.devices_hash = devices_hash # used for data augmentation

        #assert len(set(list(meta.keys())).difference(set(list(lab_hash.keys())))) == 0
        # remove keys

        self.label_hash = lab_hash

        if self.probs: # parse for data-augmentation
            label_one = []
            label_two = []

            for l in labels:
                c_label, _ = sf.read(l)  # read it all
                sess = Path(l).stem.split("-")[-1]
                # find contiguous
                tmp = self.get_segs(c_label, 1, 1)
                for s,e in tmp:
                    assert not np.where(c_label[s:e] > 1)[0].any()
                tmp = [(sess, x[0], x[1]) for x in tmp]  # we need session also
                label_one.extend(tmp)

                # do the same for two speakers
                tmp = self.get_segs(c_label, 2, 2)
                for s, e in tmp:
                    assert not np.where(c_label[s:e] != 2)[0].any()
                tmp = [(sess, x[0], x[1]) for x in tmp]
                label_two.extend(tmp)

            self.label_one = label_one
            self.label_two = label_two

        self.tot_length = int(np.sum([len(sf.SoundFile(l)) for l in labels]) / segment)

        self.set_feats_func()

        if synth:
            self.synth=synth
            # using synthetic data.

    def get_segs(self, label_vector, min_speakers,  max_speakers):

        segs = []
        label_vector =  np.logical_and(label_vector <= max_speakers, label_vector >= min_speakers)
        changePoints = np.where((label_vector[:-1] != label_vector[1:]) == True)[0]
        changePoints = np.concatenate((np.array(0).reshape(1, ), changePoints))

        if label_vector[0] == 1:
            start = 0
        else:
            start = 1
        for i in range(start, len(changePoints) - 1, 2):
            if (changePoints[i + 1] - changePoints[i]) > 30: # if only more than 30 frames
                segs.append([changePoints[i] +1, changePoints[i + 1]-1])

        return segs

    def set_feats_func(self):

        # initialize feats_function
        if self.configs["feats"]["type"] == "mfcc_kaldi":
            from torchaudio.compliance.kaldi import mfcc
            self.feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["mfcc_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "fbank_kaldi":
            from torchaudio.compliance.kaldi import fbank
            self.feats_func = lambda x: fbank(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["fbank_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "spectrogram_kaldi":
            from torchaudio.compliance.kaldi import spectrogram
            self.feats_func = lambda x: spectrogram(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **self.configs["spectrogram_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError

    def __len__(self):
        return self.tot_length

    def noaugm(self):
        # no augmentation
        file = np.random.choice(self.devices)
        sess = Path(file).stem.split(".")[0]
        start = np.random.randint(0, len(sf.SoundFile(self.label_hash[sess])) - self.segment - 2)
        stop = start + self.segment
        label, _ = sf.read(self.label_hash[sess], start=start, stop=stop)
        if self.configs["task"] == "vad":
            label = label >= 1
        elif self.configs["task"] == "osd":
            label = label >= 2
        elif self.configs["task"] == "vadosd":
            label = np.clip(label, 0, 2)
        elif self.configs["task"] == "count":
            pass
        else:
            raise EnvironmentError
        # get file start
        start = int(start * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
        stop = int(stop * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] )#+ \
                   # self.configs["data"]["fs"]* self.configs["feats"]["hop_size"] * 2)

        audio, fs = sf.read(file, start=start, stop=stop)

        if len(audio.shape) > 1:  # binaural
            audio = audio[:, np.random.randint(0, 1)]


        audio = self.feats_func(audio)
        label = label[:audio.shape[-1]]
        return audio, torch.from_numpy(label).long(), torch.ones(len(label)).bool()

    @staticmethod
    def normalize(signal, target_dB):

        fx = (AudioEffectsChain().custom(
            "norm {}".format(target_dB)))
        signal = fx(signal)
        return signal

    def __getitem__(self, item):

        if not self.probs:
            return self.noaugm()
        else:
            spkrs = np.random.choice([1, 4], p=self.probs)

            if spkrs == 1:
                return self.noaugm()
            elif spkrs ==4:
                # sample 2 from labels one
                mix = []
                labels = []
                first_lvl = None
                maxlength = None
                for i in range(spkrs):
                    sess, start, stop = random.choice(self.label_one)
                    label, _ = sf.read(self.label_hash[sess], start=start, stop=stop)

                    # get file start
                    start = int(start * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
                    stop = int(stop * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
                    file = np.random.choice(self.devices_hash[sess])
                    audio, fs = sf.read(file, start=start, stop=stop)
                    if len(audio.shape) > 1:  # binaural
                        audio = audio[:, np.random.randint(0, 1)]
                    if i == 0:
                        c_lvl = np.clip(random.normalvariate(*self.configs["augmentation"]["abs_stats"]), -30, -4) # allow for clipping  in CHiME some devices are clipped
                        first_lvl = c_lvl
                        audio = self.normalize(audio, c_lvl)
                        maxlength = len(audio)
                    else:
                        c_lvl = np.clip(first_lvl - random.normalvariate(*self.configs["augmentation"]["rel_stats"]), first_lvl-10, min(first_lvl+10, -4))
                        audio = self.normalize(audio, c_lvl)
                        rand_offset = random.randint(0, maxlength)
                        # pad only heads
                        audio = np.pad(audio, (rand_offset, 0), 'constant')
                        label = np.pad(label, (int(rand_offset /  (self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])) , 0), 'constant')
                        maxlength = max(len(audio), maxlength)

                    mix.append(audio)
                    labels.append(label)

                assert maxlength == max([len(x) for x in mix])
                if maxlength > self.segment*self.configs["data"]["fs"] * self.configs["feats"]["hop_size"]:
                    mix = [x[:int(self.segment*self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])] for x in mix]
                    labels = [x[:self.segment] for x in labels]
                    valid = torch.ones(self.segment).bool()
                else:
                    valid = torch.ones(self.segment).bool()
                    valid[int(maxlength/ (self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])):] = False

                padlen = int(self.segment * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
                mix = [np.pad(x, (0, padlen - len(x)), 'constant') for x in mix]
                mix = np.sum(np.stack(mix), 0)
                mix = np.clip(mix, -1, 1)  # clipping audio

                padlen = self.segment
                labels = [np.pad(x, (0, padlen - len(x)), 'constant') for x in labels]
                labels = np.sum(np.stack(labels), 0)
                mix = self.feats_func(mix)
                #assert mix.shape[-1] == 298
                #assert mix.shape[-1] == len(labels)
                labels = labels[:mix.shape[-1]]
                if self.configs["task"] == "vadosd":
                    labels = np.clip(labels, 0, 2)
                valid = valid[:mix.shape[-1]]
                return mix, torch.from_numpy(labels).long(), valid

            elif spkrs == 3:
                pass
                # sample 1 from label one and two from label two
            elif spkrs == 4:
                pass
                # sample 2 from label one and two from label two

class OnlineChunkedFeats(Dataset):

    def __init__(self, chime6_root, split, label_root, configs, segment=300):

        self.configs = configs
        self.segment = segment
        meta = parse_chime6(chime6_root, split)

        devices = {}
        for sess in meta.keys():
            devices[sess] = []
            for array in meta[sess]["arrays"].keys():
                devices[sess].extend(meta[sess]["arrays"][array]) # only channel 1

        labels = glob.glob(os.path.join(label_root, "*.wav"))
        lab_hash = {}

        for l in labels:
            l_sess = str(Path(l).stem).split("-")[-1]
            lab_hash[l_sess] = l

        self.lab_hash = lab_hash
        chunks = self.get_chunks(labels)

        examples = []
        for sess in chunks.keys():
            for s, e in chunks[sess]:
                for dev in devices[sess]:
                    examples.append((dev, s, e))

        self.examples = examples

        self.set_feats_func()

    def set_feats_func(self):

        # initialize feats_function
        if self.configs["feats"]["type"] == "mfcc_kaldi":
            from torchaudio.compliance.kaldi import mfcc
            self.feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **self.configs["mfcc_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "fbank_kaldi":
            from torchaudio.compliance.kaldi import fbank
            self.feats_func = lambda x: fbank(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                              **self.configs["fbank_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "spectrogram_kaldi":
            from torchaudio.compliance.kaldi import spectrogram
            self.feats_func = lambda x: spectrogram(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                                    **self.configs["spectrogram_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError

    def get_chunks(self, labels):

        chunks = {}
        chunk_size = self.configs["data"]["segment"]
        frame_shift = self.configs["data"]["segment"]

        for l in labels:
            sess = Path(l).stem.split("-")[-1]
            chunks[sess] = []
            # generate chunks for this file
            c_length = len(sf.SoundFile(l)) # get the length of the session files in samples
            for st, ed in _gen_frame_indices(
                    c_length, chunk_size, frame_shift, use_last_samples=False):
                chunks[sess].append([st, ed])
        return chunks


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        device, s, e = self.examples[item]
        sess = get_session(device)
        labelfile = self.lab_hash[sess]

        label, _ = sf.read(labelfile, start=s, stop=e)
        if self.configs["task"] == "vad":
            label = label >= 1
        elif self.configs["task"] == "osd":
            label = label >= 2
        elif self.configs["task"] == "vadosd":
            label = np.clip(label, 0, 2)
        elif self.configs["task"] == "count":
            pass
        else:
            raise EnvironmentError

        start = int(s * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
        stop = int(e * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] +
                   self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] * 2)

        audio, fs = sf.read(device, start=start, stop=stop)

        if len(audio.shape) > 1:  # binaural
            audio = audio[:, np.random.randint(0, 1)]

        audio = self.feats_func(audio)
        assert audio.shape[-1] == len(label)
        return audio, torch.from_numpy(label).long()


if __name__ == "__main__":

    import yaml
    with open("/home/sam/Projects/SSAD/egs/AMI/conf/train.yml", "r") as f:
        confs = yaml.load(f)


    a = OnlineChunkedFeats("/media/sam/bx500/amicorpus/audio/", "/home/sam/Desktop/amicorpus/labels/train/", confs)

    a = OnlineFeats("/media/sam/bx500/amicorpus/audio/",
                   "/home/sam/Desktop/amicorpus/labels/train/", confs, probs=[0., 1.0])
    from torch.utils.data import DataLoader
    for i in DataLoader(a, batch_size=3, shuffle=True):
        print(i)




