# Benchmarks

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](benchmarks.vi.md)

## Qualified measurements
| Resolution | Warm Transformer batch / 4 | Effective s/image | Images/min | Peak HBM/chip | Warm VAE batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 13.124 s | 3.281 | 18.29 | ~9.18 GB | 0.736 s |
| 768 | 15.228 s | 3.807 | 15.76 | ~14.76 GB | 0.750 s |
| 1024 | 33.542 s | 8.385 | 7.16 | ~12.53 GB | 0.782 s |

Source: [`acceptance/PRODUCTION_ACCEPTANCE.json`](../acceptance/PRODUCTION_ACCEPTANCE.json).

## What these numbers mean
The Transformer figures are **warm Transformer-stage measurements** for a four-image batch. Effective seconds/image and images/min are derived from that stage batch time. The VAE figures are warm VAE-stage batch measurements.

## What they do not mean
They are not:
- full cold-start latency;
- model download or runtime installation time;
- Text Encoder + Transformer + VAE end-to-end wall time;
- directly comparable to a GPU benchmark with a different timing scope.

## Correct comparison practice
A fair comparison should align prompt, seed, resolution, denoising steps, timing boundaries, warm/cold state, and batch semantics. Otherwise describe it as a reference rather than an apples-to-apples speedup claim.
