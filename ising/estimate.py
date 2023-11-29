import sys

import numpy as np


def T(*, N, k, a):
    l_max = np.emath.logn((k - 1), (1 + N * (k - 2) / k))
    q = np.exp(-a) * (k - 1)
    T_c = (1 - np.power(q, l_max)) / (1 - q) * k
    return T_c.real


def main():
    N = int(sys.argv[1])
    p = float(sys.argv[2])
    print(
        T(
            N=N,
            k=(N - 1) * p,
            a=int(sys.argv[3]),
        )
    )


if __name__ == "__main__":
    main()
