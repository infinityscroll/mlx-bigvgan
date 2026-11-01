# Optional inference operators

`BigVGAN.from_pretrained(..., use_precise_ops=True)` applies two inference-only
transformations **after** loading the original checkpoint:

- Replace dense transposed convolutions with nonzero polyphase stride-one
  convolutions. Supported geometry: `kernel_size = 2 * padding + stride`,
  dilation one, output padding zero, and explicit bias.
- Zero-pad dense input/output channels to multiples of 16 and slice the result
  back to the original output channels.

Both preserve the real-arithmetic operator on finite inputs. Floating-point
reduction order and backend kernel selection can change. The option leaves the
default path, global MLX precision policy, checkpoint files, anti-alias resamplers,
activations and output clipping unchanged. Apply it only for inference: the
transformed parameter layout is not a replacement pretrained checkpoint format.
Do not use it for training or reload weights into a transformed model.

```python
from mlx_bigvgan import BigVGAN
model = BigVGAN.from_pretrained("path/to/converted-model", local_files_only=True,
                               use_precise_ops=True)
```

## Phase construction

For input `x`, kernel `w`, stride `s`, padding `p`, and output phase `r`,
`y[m*s+r] = bias + sum_k x[m - (k-p-r)/s] * w[k]`, taking only taps for which
`k-p-r` is divisible by `s`. Reverse the selected taps to form a cross-correlation
kernel. Zero padding and the recorded crop produce that phase; interleaving
phases restores sample order. Phase weights are materialized once during
preparation so gather/reversal helpers do not reappear in every forward graph.
Polyphase decomposition and zero channel padding are established transformations.

## Check the simpler precision control first

In the recorded **MLX 0.32.2** control, starting a fresh process with
`MLX_ENABLE_TF32=0` requested full-FP32 convolution/matmul. That alone reduced
native-path MAE by 4737.8x on one fixed real-speech mel witness. The all-edits
advantage fell from 5011.8x under the default policy to **1.05784x** under that
request. This does not establish the arithmetic of every internal operation or
a general accuracy guarantee. That witness differs from the synthetic panel
below; it is not a matched rerun of those three cells.

The large default-policy gap is therefore not an intrinsic accuracy advantage of
polyphase algebra. Users seeking closer numerical agreement should consider the
simpler precision control first. The optional formulation also exposes static
kernel preparation without changing process-wide settings.

## Bounded historical result

This table is from the frozen Waveform Last Mile implementation adapted from
this repository's model layout. **It is not a new end-to-end benchmark of this
patch.** It used the pinned NVIDIA BigVGAN-v2 24 kHz / 100-band / 256x checkpoint,
one Apple M5 with 24 GiB unified memory, MLX 0.32.2, batch one, and synthetic
float32 mel `[1,100,F]`. Seeds were `20260917 + F`. MAE measures agreement with
PyTorch-MPS, not perceptual truth.

| Mel frames | Audio seconds | Native MAE | Rewritten MAE | MAE ratio | Native / rewritten ms | Native / rewritten peak MLX MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.171 | 1.106690e-04 | 1.134691e-07 | 975.3x | 68.145 / 65.625 | 705.91 / 686.03 |
| 32 | 0.341 | 1.907449e-04 | 1.345970e-07 | 1417.2x | 124.611 / 126.846 | 980.67 / 938.03 |
| 64 | 0.683 | 2.030354e-04 | 1.370920e-07 | 1481.0x | 268.484 / 262.392 | 1528.94 / 1442.04 |

The geometric mean MAE ratio is 1269.721408x on 0.171–0.683 seconds of
synthetic-mel output. No listening, linguistic-content, naturalness, long-form
or causal-streaming claim follows. Latency passed the registered 1.05x
**non-regression** bound; this is not material acceleration. Peak MLX allocation
counters were lower in each cell; they are not whole-process memory.

Timing used fresh native/rewritten/rewritten/native workers, three warmups and
20 synchronized samples per worker on AC power with low-power mode off. Loading,
first call, host copy and serialization were excluded. Static materialization
removed 144 indexing helpers from the earlier rewrite (4334 to 4190 primitives);
the final graph still exceeded the native graph's 4086 primitives.

## Small regression checks

Exact-rational geometry tests need no MLX, Torch, NumPy, weights, audio or download:

```sh
python3 -B -m unittest discover -s test -p 'test_precision_geometry.py' -v
```

Tiny native operator checks, in an existing Apple MLX environment:

```sh
RUN_MLX_OPERATOR_TESTS=1 python3 -B -m unittest discover -s test -p 'test_precision*.py' -v
```

These check algebra and implementation behavior. They do not reproduce the
historical timing, memory, 1270x ratio or a perceptual result. The historical source
and exact receipt paths/hashes are listed in [the compact evidence index](precision-evidence.json).
