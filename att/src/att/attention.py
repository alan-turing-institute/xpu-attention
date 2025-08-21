import torch
import numpy as np
import math

def flash_attention(q, k, v, M):
    N, d = q.size()
    print(f"{N=}, {d=}")

    # step 1
    # should be ints
    if M < (4 * d):
        raise ValueError(f"M ({M}) must be at least 4 times d ({d}) for flash attention to work efficiently.")
    
    Bc = min(math.ceil(M / (4 * d)), N)
    Br = min(math.ceil(M / (4 * d)), d)
    print(f"{Bc=}, {Br=}")

    # step 2
    O = torch.zeros(N,d)
    l = torch.zeros(N)
    m = torch.full((N,), fill_value=-np.inf)

    # step 3/4
    # should be ints
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    # step 5
    for j in range(0, Tc):
        # for each j, we move forward Bc rows
        Kj = k[j*Bc:(j+1)*Bc]
        Vj = v[j*Bc:(j+1)*Bc]

        for i in range(0, Tr):
            # for each i, we move forward Br rows
            Qi = q[i*Br:(i+1)*Br]
            Oi = O[i*Br:(i+1)*Br]
            li = l[i*Br:(i+1)*Br]
            mi = m[i*Br:(i+1)*Br]

            Sij = torch.matmul(Qi, Kj.T)

            mij = torch.max(Sij, dim=1).values #dim=1 is rows
            
            Pij = torch.exp(Sij - mij[:, None])  # mij needs to go from (Br,) -> (Br, 1)

            lij = torch.sum(Pij, dim=1)
            
            mi_new = torch.max(mi, mij)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij- mi_new) * lij

            # li has size (Br,) -> do same as above to convert to (Br, 1)
            # mi and mi_new have size (Br,) -> do same as above to convert to (Br, 1)
            # Oi has size (Br, d)
            # PijVj has size (Br, Bc) x (Bc, d) = (Br, d)

            Oi_numerator = li[:, None] * torch.exp(mi - mi_new)[:, None] * Oi + torch.exp(mij - mi_new)[:, None] * torch.matmul(Pij, Vj)

            # li_new has size (Br,) -> do same as above to convert to (Br, 1)
            Oi_denominator = li_new[:, None]

            # Oi_new has size (Br, d)
            Oi_new = Oi_numerator / Oi_denominator

            # write back to the actual O, l, m
            O[i*Br:(i+1)*Br] = Oi_new
            l[i*Br:(i+1)*Br] = li_new
            m[i*Br:(i+1)*Br] = mi_new

    return O

#     return torch.tensor([
#         [0.6648, 0.8648],
#         [0.6722, 0.8722],
#         [0.6796, 0.8796],
#         [0.6869, 0.8869],
#         [0.6942, 0.8942],
#         [0.7014, 0.9014],
#         [0.7086, 0.9086],
#         [0.7157, 0.9157]])
