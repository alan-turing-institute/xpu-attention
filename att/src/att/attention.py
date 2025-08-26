import math

import torch


def debug_print(output):
    # print(output)
    pass


def flash_attention(Q, K, V):
    debug_print(f"Q: {Q}")
    debug_print(f"K: {K}")
    debug_print(f"V: {V}")

    N, d = Q.shape
    M = 36
    Bc = min(math.ceil(M / (4 * d)), N)
    Br = min(math.ceil(M / (4 * d)), d)

    O = torch.zeros((N, d))
    l = torch.zeros(N)
    m = torch.ones(N) * -math.inf
    debug_print(f"N: {N}")
    debug_print(f"d: {d}")
    debug_print(f"M: {M}")
    debug_print(f"Bc: {Bc}")
    debug_print(f"Br: {Br}")
    debug_print(f"O: {O}")
    debug_print(f"l: {l}")
    debug_print(f"m: {m}")

    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    debug_print(f"Tc: {Tc}")
    debug_print(f"Tr: {Tr}")

    Qi = torch.tensor_split(Q, Tr)
    debug_print(f"Qi: {Qi}")
    assert Qi[0].size() == (Br, d)

    sizes = [Bc] * (Tc - 1) + [N % Bc or Bc]
    Ki = torch.split(K, sizes)
    debug_print(f"Ki: {Ki}")
    assert [x.size() for x in Ki] == [(x, d) for x in sizes]

    Vi = torch.split(V, sizes)
    debug_print(f"Vi: {Vi}")
    assert [x.size() for x in Vi] == [(x, d) for x in sizes]

    sizes = [Br] * (Tr - 1) + [N % Br or Br]
    Oi = torch.split(O, sizes)
    debug_print(f"Oi: {Oi}")
    assert [x.size() for x in Oi] == [(x, d) for x in sizes]

    li = torch.split(l, sizes)
    debug_print(f"li: {li}")
    assert [x.size() for x in li] == [(x,) for x in sizes]

    mi = torch.split(m, sizes)
    debug_print(f"mi: {mi}")
    assert [x.size() for x in mi] == [(x,) for x in sizes]

    for j in range(Tc):
        # Load Ki[j], Vi[j] from HBM to on-chip SRAM
        for i in range(Tr):
            # Load Qi[i], Oi[i], li[i] and mi[i] from HBM to on-chip SRAM
            Sij = Qi[i] @ Ki[j].transpose(-2, -1)
            assert (Sij.size() == (Br, Bc)) or (
                Sij.size() == (N % Br or Br, N % Bc or Bc)
            )

            m_ij = torch.max(Sij, 1).values
            assert m_ij.size() == (Br,)

            P_ij = torch.exp(Sij - m_ij.unsqueeze(1))
            assert (P_ij.size() == (Br, Bc)) or (
                P_ij.size() == (N % Br or Br, N % Bc or Bc)
            )

            l_ij = torch.sum(P_ij, 1)
            assert l_ij.size() == (Br,)

            m_ij = torch.squeeze(m_ij)
            mnewi = torch.max(mi[i], m_ij)
            assert mnewi.size() == (Br,)

            lnewi = (torch.exp(mi[i] - mnewi) * li[i]) + (
                torch.exp(m_ij - mnewi) * l_ij
            )
            assert lnewi.size() == (Br,)

            Oi[i][:] = torch.diag(torch.reciprocal(lnewi)) @ (
                (torch.diag(li[i]) @ torch.exp(mi[i] - mnewi) * Oi[i])
                + (torch.exp(m_ij - mnewi).unsqueeze(1) * P_ij @ Vi[j])
            )
            assert Oi[i].size() == (Br, d)

            li[i][:] = lnewi

            mi[i][:] = mnewi

    O = torch.cat(Oi, 0)
    assert O.size() == (N, d)
    debug_print(f"O: {O}")

    # The correct value for the test input
    # torch.tensor(
    #     [
    #         [0.6648, 0.8648],
    #         [0.6722, 0.8722],
    #         [0.6796, 0.8796],
    #         [0.6869, 0.8869],
    #         [0.6942, 0.8942],
    #         [0.7014, 0.9014],
    #         [0.7086, 0.9086],
    #         [0.7157, 0.9157],
    #     ]
    # )

    return O
