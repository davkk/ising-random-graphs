import sys

import numpy as np


def estimate_critical_temperature(*, n, p, alpha):
    k = n * p

    T_c = 0
    used = 0
    r = 1

    while used < n:
        con = min(n - used, k * np.power(k - 1, r - 1))

        used += con
        T_c += con * np.exp(-alpha * (r - 1))

        r += 1

    return T_c


def main():
    print(
        estimate_critical_temperature(
            n=int(sys.argv[1]),
            p=float(sys.argv[2]),
            alpha=int(sys.argv[3])
        )
    )


if __name__ == "__main__":
    main()
