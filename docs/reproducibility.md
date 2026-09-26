# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](reproducibility.vi.md)

## Accepted baseline
```text
accelerator   Kaggle TPU v5e-8
devices       8
topology      4x2
steps         4
seeds         42,43,44,45
resolutions   512,768,1024
```

## Authority chain
1. Runtime source is pinned by the v1.0.0 qualified commit.
2. Converted model artifacts are supplied through public model distribution.
3. The production runner resolves the qualified attention policy.
4. Text Encoder, Transformer, and VAE execute as isolated stages.
5. Acceptance checks deterministic warm reruns and unique per-seed outputs.
6. Final PNGs are compared with previously visually accepted qualification outputs.

## Machine-readable authority
[`acceptance/PRODUCTION_ACCEPTANCE.json`](../acceptance/PRODUCTION_ACCEPTANCE.json) contains policy, timing, HBM, and accepted PNG SHA-256 values for each resolution.

## What invalidates equivalence
Treat a run as a new experiment if you change JAX/Keras versions, Transformer source, BF16 timestep semantics, sharding/topology, attention policy, denoising steps, checkpoint mapping, VAE binding, or resolution outside the qualified set.

CPU tests can detect several source-contract regressions, but TPU qualification remains the hardware authority.
