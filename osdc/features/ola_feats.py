import torch
import numpy as np
from SSAD.utils.oladd import _gen_frame_indices


def compute_feats_windowed(feats_func, audio, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        tmp = feats_func(audio[st:ed])
        #tmp = tmp if st == 0 else tmp[:, 160*10:]
        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out



if __name__ == "__main__":

    from gammatone import get_gammatonegm_

    random = np.random.random((16000*30))

    gammas = lambda x : get_gammatonegm_(x)
    windowed = compute_feats_windowed(gammas, random, 16000*4, 16000*4-160*2)

    direct = gammas(random)

    np.testing.assert_array_almost_equal(direct, windowed)