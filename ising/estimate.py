import sys

import numpy as np


def T(*, N, k, a):
    l_max = np.emath.logn((k - 1), (1 + N * (k - 2) / k))
    q = np.exp(-a) * (k - 1)
    return (1 - np.power(q, l_max)) / (1 - q) * k


# def estimate_critical_temperature(*, n, p, alpha):
#     k = n * p
#
#     T_c = 0
#     used = 0
#     r = 1
#
#     while used < n:
#         con = min(n - used, k * np.power(k - 1, r - 1))
#
#         used += con
#         T_c += con * np.exp(-alpha * (r - 1))
#
#         r += 1
#
#     return T_c


def main():
    print(
        T(
            n=int(sys.argv[1]),
            p=float(sys.argv[2]),
            alpha=int(sys.argv[3]),
        )
    )


if __name__ == "__main__":
    main()
