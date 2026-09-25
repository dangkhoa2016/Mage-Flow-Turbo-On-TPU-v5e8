# Mage-Flow Turbo on TPU v5e-8

Keras 3 / JAX runtime source and production orchestration for Mage-Flow Turbo on Kaggle TPU v5e-8.

The qualified production profile uses eight TPU devices with a preferred **4x2 replica/model topology**, four denoising steps, segmented attention at 512/768, and segmented query-chunk attention at 1024.

## Production profile

| Resolution | Attention | Query chunk | Warm transformer batch | Images/min |
|---|---|---:|---:|---:|
| 512 | segmented | — | 13.124 s / 4 | 18.29 |
| 768 | segmented | — | 15.228 s / 4 | 15.76 |
| 1024 | segmented-query-chunk | 256 | 33.542 s / 4 | 7.16 |

The acceptance run also verified 397 transformer leaves, 174 sharded / 223 replicated parameters, VAE binding coverage 728/728, bit-exact warm reruns, unique per-seed outputs, and PNG byte identity with the visually accepted qualification runs.

## Repository layout

```text
runtime/final/source/mage_flow_keras/  qualified JAX/Keras transformer source
bootstrap/                             production TPU runner and stage helpers
tests/                                 CPU regression tests
docs/                                  English/Vietnamese production notes
acceptance/                            compact production acceptance metadata
```

## Runtime prerequisites

Large model checkpoints, the pinned Python runtime cache, and the external `experiment` runtime support package are intentionally not stored in this repository.

The production runner accepts those paths explicitly through `--runtime-root`, optional `--runtime-site`, checkpoint arguments, RoPE basis inputs, and the canonical model root. This keeps the Git repository reviewable while preserving reproducible runtime contracts.

See [TPU v5e-8 production notes](docs/tpu-v5e8-production.md) for the qualified policy, benchmark data, correctness constraints, and release-verification requirements.
