import numpy as np

N = 10000
p = 0.004
k = N * p
r_max = int(np.ceil(np.emath.logn(k, N)))
alpha = 1

T_c = 0
used = 0

for r in range(1, r_max + 1):
    exp = np.exp(-alpha * (r - 1))
    con = k * np.power(k - 1, r - 1)

    T_c += min(N - used, con) * exp
    used += con

    if used > N:
        break

print(T_c)
