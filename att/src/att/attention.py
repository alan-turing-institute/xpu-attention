import torch
import numpy as np

def flash_attention(q, k, v):
    N, d = q.size()

    M = 4
    d = 1

    # step 1
    Bc = M / (4 * d)
    Br = min((M / (4 * d)), d)

    # step 2
    O = torch.zeros(N,d)
    l = torch.zeros(N)
    m = torch.full(N, fill_value=-np.inf)

    # step 3/4
    Tr = N / Br
    Tc = N / Bc

    # step 5
    for j in range(0, Tc):
        Kj = k[j:j+Tc]
        Vj = v[j:j+Tc]

        for i in range(0, Tr):
            Qi = q[i:i+Tr]
            Oi = O[i:i+Tr]
            li = l[i:i+Tr]
            mi = m[i:i+Tr]
            
            Sij = torch.matmul(Qi, Kj.T)
            assert Sij.size() == (Br, Bc), f"Expected Sij size (Br, Bc), got {Sij.size()}"

            mij = torch.max(Sij, dim=1) #dim=1 is rows
            assert mij.size() == (Br,), f"Expected mij size (Br,), got {mij.size()}"
            
            Pij = torch.exp(Sij - mij)
            assert Pij.size() == (Br, Bc), f"Expected Pij size (Br, Bc), got {Pij.size()}"

            lij = torch.sum(Pij, dim=1)
            
            mi_new = max(mi, mij)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij- mi_new) * lij

            Oi = (torch.diag(li) * torch.exp(mi - mi_new) * Oi + torch.exp(mij - mi_new) * torch.matmul(Pij, Vj)) / torch.diag(li_new)

            li = li_new
            mi = mi_new
        
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
