import sys

import numpy as np

from . import ising


def main():
    ising.simulate(
        np.load(sys.argv[1]),
        np.uint32(sys.argv[2]),
        np.float64(sys.argv[3]),
    )


if __name__ == "__main__":
    main()
