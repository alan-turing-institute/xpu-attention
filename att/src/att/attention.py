import math

import torch


def flash_attention(Q, K, V):
    print(f"Q: {Q}")
    print(f"K: {K}")
    print(f"V: {V}")

    N, d = Q.shape
    M = 16
    Bc = min(math.floor(M / (4 * d)), N)
    Br = min(math.floor(M / (4 * d)), d)

    O = torch.zeros((N, d))
    l = torch.zeros(N)
    m = torch.ones(N) * -math.inf
    print(f"N: {N}")
    print(f"d: {d}")
    print(f"M: {M}")
    print(f"Bc: {Bc}")
    print(f"Br: {Br}")
    print(f"O: {O}")
    print(f"l: {l}")
    print(f"m: {m}")

    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    print(f"Tc: {Tc}")
    print(f"Tr: {Tr}")

    Qi = torch.tensor_split(Q, Tr)
    print(f"Qi: {Qi}")
    assert Qi[0].size() == (Br, d)

    Ki = torch.tensor_split(K, Tc)
    print(f"Ki: {Ki}")
    assert Ki[0].size() == (Bc, d)

    Vi = torch.tensor_split(V, Tc)
    print(f"Vi: {Vi}")
    assert Vi[0].size() == (Bc, d)

    Oi = torch.tensor_split(O, Tr)
    print(f"Oi: {Oi}")
    assert Oi[0].size() == (Br, d)
    Oi = list(Oi)

    li = torch.tensor_split(l, Tr)
    print(f"li: {li}")
    assert li[0].size() == (Br,)
    li = list(li)

    mi = torch.tensor_split(l, Tr)
    print(f"mi: {mi}")
    assert mi[0].size() == (Br,)
    mi = list(mi)

    for j in range(Tc):
        # Load Ki[j], Vi[j] from HBM to on-chip SRAM
        for i in range(Tr):
            # Load Qi[i], Oi[i], li[i] and mi[i] from HBM to on-chip SRAM
            Sij = Qi[i] @ Ki[j].transpose(-2, -1)
            assert Sij.size() == (Br, Bc)

            m_ij = torch.squeeze(torch.max(Sij, 1, keepdim=True).values)
            assert m_ij.size() == (Br,)

            P_ij = torch.exp(Sij - m_ij)
            assert P_ij.size() == (Br, Bc)

            l_ij = torch.squeeze(torch.sum(P_ij, 1, keepdim=True))
            assert l_ij.size() == (Br,)

            mnewi = torch.max(mi[i], m_ij)
            assert mnewi.size() == (Br,)

            lnewi = (torch.exp(mi[i] - mnewi) * li[i]) + (
                torch.exp(m_ij - mnewi) * l_ij
            )
            assert lnewi.size() == (Br,)

            Oi[i] = torch.diag(torch.reciprocal(lnewi)) @ (
                (torch.diag(li[i]) @ torch.exp(mi[i] - mnewi) * Oi[i])
                + (torch.exp(m_ij - mnewi) * P_ij @ Vi[j])
            )
            assert Oi[i].size() == (Br, d)

            li[i] = lnewi

            mi[i] = mnewi

    O = torch.cat(Oi, 0)
    assert O.size() == (N, d)
    print(f"O: {O}")

    return O

    # The correct value
    return torch.tensor(
        [
            [0.6648, 0.8648],
            [0.6722, 0.8722],
            [0.6796, 0.8796],
            [0.6869, 0.8869],
            [0.6942, 0.8942],
            [0.7014, 0.9014],
            [0.7086, 0.9086],
            [0.7157, 0.9157],
        ]
    )
