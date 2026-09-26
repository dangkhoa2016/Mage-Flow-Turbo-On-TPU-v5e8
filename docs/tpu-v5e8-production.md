# Mage-Flow TPU v5e-8 Production Runtime

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](tpu-v5e8-production.vi.md)

This document records the production runtime that passed acceptance on 2026-09-25.

## Qualified contract

- Accelerator: Kaggle TPU v5e-8, 8 devices.
- Preferred topology: 4x2.
- Denoising steps: 4.
- Default seeds for 4x2 acceptance: 42, 43, 44, 45.
- Transformer parameters: 397 leaves; 174 sharded and 223 replicated.
- VAE runtime binding: 728/728.
- Production source: `runtime/final/source/mage_flow_keras`.

## Attention policy

| Resolution | Production attention | Query chunk |
|---|---|---:|
| 512 | segmented | n/a |
| 768 | segmented | n/a |
| 1024 | segmented-query-chunk | 256 |

Segmented production modes intentionally require equal packed request lengths and fail closed otherwise.

## Production benchmark authority

| Resolution | Warm transformer batch | Effective s/image | Images/min | Peak HBM/chip | Warm VAE batch |
|---|---:|---:|---:|---:|---:|
| 512 | 13.124 s / 4 | 3.281 | 18.29 | ~9.18 GB | 0.736 s |
| 768 | 15.228 s / 4 | 3.807 | 15.76 | ~14.76 GB | 0.750 s |
| 1024 | 33.542 s / 4 | 8.385 | 7.16 | ~12.53 GB | 0.782 s |

These figures come from the final promoted production runner, not exploratory scripts.

> These are warm stage-level measurements. They are not full cold-start or end-to-end image-generation latency.

## Correctness requirements

Do not regress BF16 timestep semantics: `jnp.asarray(timesteps, dtype=jnp.bfloat16).astype(jnp.float32)`.

Construct and restore model state on CPU before TPU mesh `device_put`. Keep per-request segmented attention for concurrent execution and query chunk 256 at 1024 unless a new TPU qualification replaces this authority.

The final production outputs at 512, 768, and 1024 were byte-identical to the previously visually accepted qualification PNGs.

## Runner

Use `bootstrap/06_run_tpu_inference.py` with isolated text-encoder, transformer, and VAE subprocess stages.

The runner accepts `1x8`, `2x4`, and `4x2` topologies. Only the qualified 4-step schedule is accepted by the production runner.

## Release verification

CPU tests and CI cover repository integration; they do not replace TPU release verification. A runtime-semantic release candidate should be rerun on TPU v5e-8 at 4x2 for 512, 768, and 1024 and compared against the accepted evidence authority.

Evidence archive SHA-256: `4982c750914599561cd5255c1ad9e58cfb02a9f2c26a299801271a92d2c0bf68`.

See [Release and verification](release-and-verification.md) and [Benchmarks](benchmarks.md).
