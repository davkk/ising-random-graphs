import numpy as np

N = 1000
print(f"{N=}")
p = 0.05
print(f"{p=}")

k = N * p
print(f"{k=}")

r_max = int(np.ceil(np.emath.logn(k, N)))
print(f"{r_max=}")

alpha = 1
print(f"{alpha=}")

T_c = 0
used = 0
for r in range(1, r_max + 1):
    exp = np.exp(-alpha * (r - 1))
    con = k * np.power(k - 1, r - 1)

    T_c += min(N - used, con) * exp
    used += con

    if used > N:
        break

print(f"{T_c=}")
