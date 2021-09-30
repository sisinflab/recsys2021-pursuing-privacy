import random

Q = 121639451781281043402593


def encrypt(x, n_shares=2):
    shares = list()
    for i in range(n_shares - 1):
        shares.append(random.randint(0, Q))
    # final_share = Q - (sum(shares) % Q) + x
    final_share = x - (sum(shares) % Q)
    shares.append(final_share)
    return tuple(shares)


def decrypt(shares):
    return sum(shares) % Q


secrets = [100, 200, 300]
local_shares = [[], [], []]

for s in secrets:
    t = encrypt(s, 3)
    for i in range(len(t)):
        local_shares[i].append(t[i])

local_computation = [sum(u) % Q for u in local_shares]
# The values in local_computation represent the shares of (A+B+C)
print(local_computation)
print(decrypt(local_computation) / len(local_computation))


# We now integrate the SPDZ protocol, which assumes the presence of a cryptoprovider

def generate_mul_triple():
    a = random.randrange(Q)
    b = random.randrange(Q)
    a_mul_b = (a * b) % Q
    return encrypt(a), encrypt(b), encrypt(a_mul_b)

