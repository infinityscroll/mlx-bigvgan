"""Explicitly opt-in, tiny operator checks on an installed Apple MLX runtime."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_precision_geometry import scatter_reference


@unittest.skipUnless(os.environ.get('RUN_MLX_OPERATOR_TESTS') == '1', 'set RUN_MLX_OPERATOR_TESTS=1 on Apple MLX')
class MLXTests(unittest.TestCase):
    def test_actual_polyphase_operator_against_independent_scatter(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_bigvgan.precision import PhaseConvTranspose1d
        for kernel, stride, padding in ((4, 2, 1), (8, 4, 2), (16, 4, 6)):
            with self.subTest(kernel=kernel, stride=stride):
                value = [[[(n * 3 + c - 10) / 32 for c in range(3)] for n in range(7)]]
                weight = [[[(o - k + c) / 64 for c in range(3)] for k in range(kernel)] for o in range(5)]
                bias = [o / 32 for o in range(5)]
                layer = nn.ConvTranspose1d(3, 5, kernel, stride=stride, padding=padding)
                layer.weight = mx.array(weight, dtype=mx.float32); layer.bias = mx.array(bias, dtype=mx.float32)
                operator = PhaseConvTranspose1d(layer)
                actual = operator(mx.array(value, dtype=mx.float32)).tolist()
                expected = scatter_reference(value, weight, bias, stride, padding)
                for a_batch, e_batch in zip(actual, expected, strict=True):
                    for a_row, e_row in zip(a_batch, e_batch, strict=True):
                        for a, e in zip(a_row, e_row, strict=True): self.assertAlmostEqual(a, e, delta=5e-6)
                self.assertEqual(len(operator.phase_weights), stride)

    def test_dense_padding_against_direct_correlation(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_bigvgan.precision import PreciseConv1d
        value = [[[(n + c - 5) / 16 for c in range(3)] for n in range(7)]]
        weight = [[[(o - k + c) / 32 for c in range(3)] for k in range(3)] for o in range(5)]
        layer = nn.Conv1d(3, 5, 3, padding=1)
        layer.weight = mx.array(weight, dtype=mx.float32); layer.bias = mx.array([o / 16 for o in range(5)], dtype=mx.float32)
        actual = PreciseConv1d(layer)(mx.array(value, dtype=mx.float32)).tolist()[0]
        for n, row in enumerate(actual):
            for o, measured in enumerate(row):
                expected = o / 16 + sum(value[0][n+k-1][c] * weight[o][k][c] for k in range(3) if 0 <= n+k-1 < 7 for c in range(3))
                self.assertAlmostEqual(measured, expected, delta=5e-6)


if __name__ == '__main__': unittest.main()
