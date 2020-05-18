import intervals as I
import numpy as np

def delete_overlapping(result):
    """

    :param result: result is a list sorted list of [[start, stop], [start, stop]]
    :return: list with no overlapped segments
    """
    stack = [I.closed(*result[0])]

    for indx in range(1, len(result)):

        current = I.closed(*result[indx]).difference(stack[-1])

        if not current.is_empty():
            stack.append(current)

    return stack


def merge_intervals(intervals, delta= 0.0):
    """
    A simple algorithm can be used:
    1. Sort the intervals in increasing order
    2. Push the first interval on the stack
    3. Iterate through intervals and for each one compare current interval
       with the top of the stack and:
       A. If current interval does not overlap, push on to stack
       B. If current interval does overlap, merge both intervals in to one
          and push on to stack
    4. At the end return stack
    """
    if not intervals:
        return intervals
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def apply_collar(segments, collar=0.0):

    tmp = []
    for s,e in segments:
        s -= collar
        e +=collar
        tmp.append([s, e])

    return tmp

def delete_shorter(segments, th=np.inf):

    tmp = []
    for s, e in segments:
        if (e-s) > th:
            tmp.append([s, e])
    return tmp


def time_to_samples(x, sample_rate=16000):
    hours, mins, sec = x.split(":")
    hours = int(hours)
    mins = int(mins)
    sec, dec = sec.split(".")
    sec = int(sec)
    dec = int(dec)

    return int((hours * 3600 + mins * 60 + sec + dec * 0.01)*sample_rate)