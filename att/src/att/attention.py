import math

import torch


def flash_attention(q, k, v):
    """
    Flash Attention implementation using tiling for memory efficiency.

    Args:
        q: Query tensor of shape (N, d)
        k: Key tensor of shape (N, d)
        v: Value tensor of shape (N, d)

    Returns:
        Output tensor of shape (N, d)
    """
    N, d = q.shape

    # Block size for tiling (can be tuned based on memory constraints)
    block_size = min(64, N)

    # Initialize output and statistics
    O = torch.zeros_like(q)
    l = torch.zeros(N, 1)  # row sums
    m = torch.full((N, 1), -float("inf"))  # row maxes

    # Process blocks
    for j in range(0, N, block_size):
        j_end = min(j + block_size, N)

        # Load current blocks
        K_j = k[j:j_end]  # (block_size, d)
        V_j = v[j:j_end]  # (block_size, d)

        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)

            # Load query block
            Q_i = q[i:i_end]  # (block_size, d)

            # Compute attention scores
            S_ij = (Q_i @ K_j.T) / math.sqrt(d)  # (block_size, block_size)

            # Update statistics
            m_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]  # (block_size, 1)
            m_i_new = torch.maximum(m[i:i_end], m_ij)

            # Compute probability updates
            P_ij = torch.exp(S_ij - m_i_new)  # (block_size, block_size)
            l_ij = torch.sum(P_ij, dim=-1, keepdim=True)  # (block_size, 1)

            # Update output and statistics
            alpha = torch.exp(m[i:i_end] - m_i_new)
            O[i:i_end] = alpha * O[i:i_end] + P_ij @ V_j
            l[i:i_end] = alpha * l[i:i_end] + l_ij
            m[i:i_end] = m_i_new

    # Final normalization
    O = O / l

    return O
