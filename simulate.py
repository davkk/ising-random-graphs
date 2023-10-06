import sys

import numpy as np

import ising

ising.simulate(
    np.load(sys.argv[1]),
    np.uint32(sys.argv[2]),
    np.float32(sys.argv[3]),
)
