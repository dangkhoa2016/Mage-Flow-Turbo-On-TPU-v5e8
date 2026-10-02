# Inference Guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](inference-guide.vi.md)

The production entry point is `bootstrap/06_run_tpu_inference.py`.

## Core arguments
| Argument | Contract |
| --- | --- |
| `--topology` | `1x8`, `2x4`, or `4x2`; qualification uses `4x2` |
| `--resolution` | 512, 768, or 1024 |
| `--seeds` | comma-separated seed list |
| `--steps` | qualified runner requires 4 |
| `--attention` | `auto`, `masked-global`, `segmented`, `segmented-query-chunk` |
| `--query-chunk` | positive integer; qualified 1024 value is 256 |
| `--model-root` | canonical model root |
| `--runtime-root` | runtime support root |
| `--runtime-site` | optional isolated Python site |
| checkpoint args | explicit Text Encoder, Transformer, and VAE checkpoints |
| RoPE args | explicit dimension-16 and dimension-56 basis assets |
| `--output` | output directory |

## Fail-closed behavior
The runner rejects a denoising schedule other than four steps. Runtime attention code rejects unknown modes, non-positive query chunk values, unsupported attention kwargs, and unequal packed request lengths for segmented production modes.

This is deliberate: unsupported configurations should fail visibly instead of silently producing an unqualified result.

## Recommended production settings
```text
topology    4x2
steps       4
attention   auto
seeds       42,43,44,45 (acceptance baseline)
```

Use [Reproducibility](reproducibility.md) when reproducing the accepted baseline rather than merely experimenting with the runtime.
