import numpy as np


def splitOtherStem(left, right, sample_rate):
    """
    Simple heuristic-based instrument separation from 'other' stem.

    Input:
        left, right: numpy arrays (stereo channels)
        sample_rate: int

    Output:
        dict of separated instruments with stereo channels
    """

    length = len(left)

    guitarL = np.zeros(length)
    guitarR = np.zeros(length)

    pianoL = np.zeros(length)
    pianoR = np.zeros(length)

    synthL = np.zeros(length)
    synthR = np.zeros(length)

    for i in range(length):
        l = left[i]
        r = right[i]

        energy = abs(l) + abs(r)

        # 🎸 Guitar (mid transient energy)
        if 0.02 < energy < 0.3:
            guitarL[i] = l * 0.8
            guitarR[i] = r * 0.8

        # 🎹 Piano (balanced harmonic energy)
        elif 0.3 <= energy < 0.6:
            pianoL[i] = l * 0.7
            pianoR[i] = r * 0.7

        # 🎛️ Synth (sustained / high energy)
        elif energy >= 0.6:
            synthL[i] = l * 0.9
            synthR[i] = r * 0.9

    return {
        "guitar": {
            "left": guitarL,
            "right": guitarR
        },
        "piano": {
            "left": pianoL,
            "right": pianoR
        },
        "synth": {
            "left": synthL,
            "right": synthR
        }
    }