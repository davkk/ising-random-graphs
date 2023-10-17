import fileinput
from collections import defaultdict

import numpy as np


def main():
    with fileinput.input() as f:
        energies = defaultdict(list)
        magnets = defaultdict(list)

        for line in f:
            temp, _, energy, magnet = map(np.float64, line.split())

            energies[temp].append(energy)
            magnets[temp].append(magnet)

        for temp in energies.keys():
            print(temp, np.average(energies[temp]), np.average(magnets[temp]))


if __name__ == "__main__":
    main()
