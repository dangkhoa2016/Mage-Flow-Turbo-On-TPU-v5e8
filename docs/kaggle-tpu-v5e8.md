# Kaggle TPU v5e-8 Guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle-tpu-v5e8.vi.md)

## Qualified environment
The formal production qualification targets Kaggle TPU v5e-8 with 8 TPU devices. The published profile uses topology `4x2`, four denoising steps, and seeds 42–45 for acceptance.

## Recommended public path
For the shortest reproducible path, use the public [Kaggle TPU demo](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8-demo) with the [Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8) attached.

The GitHub runner is primarily the source/audit surface. It expects explicit paths to model checkpoints, runtime support, VAE manifest, and RoPE basis assets.

## Preflight expectations
Before inference, verify:
- JAX sees TPU as the backend;
- exactly 8 TPU devices are available;
- the requested topology is compatible with 8 devices;
- required model/runtime paths exist;
- resolution is 512, 768, or 1024;
- `--steps` is exactly 4 for the qualified runner.

A session that silently falls back to CPU should not be treated as a successful TPU reproduction.

## Production policy
Use `--attention auto` unless deliberately reproducing a lower-level diagnostic. Auto selects the qualified resolution-specific attention mode.

CPU CI is useful before a TPU run, but it does not establish TPU performance or end-to-end acceptance.
