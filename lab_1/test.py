import numpy as np


def get_tuple():
    return 1, 2, 2

a = np.zeros((4, 4, 4, 4))

a[1, 2, 2, 1] = 1

i, j, k = get_tuple()
print(a[get_tuple()][1])
print(a[i, j, k, 1])
print(a[1, 2, 2, 0])