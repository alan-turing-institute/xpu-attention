import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

# import pdb
from att import attention


def forward(q, k, v):
    """Adapted from minGPT."""
    # causal self-attention; Self-attend: (N, d) x (d, N) -> (N, N)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (N, N) x (N, d) -> (N, d)
    return y


class TestAttention(unittest.TestCase):

    q = torch.tensor(
        [
            [0.1000, 0.3000],
            [0.2000, 0.4000],
            [0.3000, 0.5000],
            [0.4000, 0.6000],
            [0.5000, 0.7000],
            [0.6000, 0.8000],
            [0.7000, 0.9000],
            [0.8000, 1.0000],
        ]
    )
    k = q + 0.1
    v = q + 0.2

    def test_forward(self):
        """That Karpathy's attention is sensible."""

        result = forward(self.q, self.k, self.v)

        expected = torch.tensor(
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

        self.assertEqual(result.shape, self.v.shape)
        self.assertTrue(np.allclose(result, expected, rtol=1e-1))

    def test_attention_m32(self):
        actual = attention.flash_attention(self.k, self.q, self.v, M=32)
        expected = forward(self.k, self.q, self.v)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-1))

    def test_attention_m36(self):
        actual = attention.flash_attention(self.k, self.q, self.v, M=36)
        expected = forward(self.k, self.q, self.v)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
