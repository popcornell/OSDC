import numpy as np
from scipy.signal import get_window

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


def overlap_add(tensor, function, window_size=1600 * 6, stride=1600 * 3, win_type='hanning', pad_left=True,
                n_classes=0, lookahead=1600*2, lookbehind=1600):
    # tensor assumed to be B, C , T

    assert len(tensor.shape) >= 2

    if window_size // stride != 2 or not pad_left:
        raise NotImplementedError

    orig_length = tensor.shape[-1] # original length
    if pad_left:
        pad_left = stride
    else:
        raise NotImplementedError

    pad_right = stride + lookahead # always pad the end
    n, r = divmod(orig_length + pad_right, stride)
    if r != 0:
        n = n+1
        pad_right += stride*n - (orig_length + pad_right)

    pad_dims = [(0,0) for x in range(len(tensor.shape[1:]))]
    npad = (*pad_dims, (pad_left, pad_right))
    tensor = np.pad(tensor, npad, mode="constant")
    window = get_window(win_type, window_size)
    # make window same dimension as tensor channels
    reshape_dims = [1]*len(tensor.shape[:-1])
    window = window.reshape((*reshape_dims, -1))

    if n_classes:
        b, ch, t = window
        window = window # TODO
        raise NotImplementedError

    # first segment
    temp = function(tensor[..., :window_size + lookahead])[..., :window_size] * window

    result = [] # will be discarded
    buff = temp[..., stride:window_size]

    for i in range(1, n):
        temp = np.copy(tensor[..., i*stride - lookbehind: i*stride+ window_size + lookahead][..., lookbehind:window_size + lookbehind])
        temp = function(temp)
        temp *= window
        result.append(temp[..., :stride] + buff)
        buff = temp[..., stride:window_size]

    result = np.concatenate(result, -1)

    return result[..., :orig_length]



def sequential_predict(tensor, function, left_context ,right_context ):

    assert len(tensor.shape) >= 2

    pad_right = right_context
    pad_left = left_context

    orig_len = tensor.shape[-1]

    pad_dims = [(0, 0) for x in range(len(tensor.shape[1:]))]
    npad = (*pad_dims, (pad_left, pad_right))
    tensor = np.pad(tensor, npad, mode="constant")



    # first segment
    temp = function(tensor[..., : left_context + right_context +1])
    result = [temp]
    for i in range(1, orig_len):
        temp = np.copy(tensor[..., i: i+left_context + right_context +1 ])
        temp = function(temp)
        result.append(temp)

    result = np.concatenate(result, -1)

    return result[..., :orig_len]


if __name__ == "__main__":

    input = np.random.random((3, 64, 10000))
    f = lambda x : x[:, :, 21:-20]

    res = sequential_predict(input, f, 21, 20)
    #res = overlap_add(input, f, window_size=500,stride=1, lookbehind=0)
    np.testing.assert_array_almost_equal(res, input)