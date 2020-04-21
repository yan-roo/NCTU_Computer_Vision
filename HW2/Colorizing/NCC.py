import numpy as np


def ncc(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))


def align(a, b, t):
    max_ncc = -1
    max_ncc_shift = [0, 0]
    i_value = np.linspace(-t, t, 2*t, dtype=int)
    j_value = np.linspace(-t, t, 2*t, dtype=int)
    for i in i_value:
        for j in j_value:
            ncc_num = ncc(a, np.roll(b, [i, j], axis=(0,1)))
            if ncc_num > max_ncc:
                max_ncc = ncc_num
                max_ncc_shift = [i, j]
    return max_ncc_shift
