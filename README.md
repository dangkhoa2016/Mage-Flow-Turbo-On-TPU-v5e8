# Mage-Flow-Turbo on TPU v5e-8

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8?display_name=tag&sort=semver)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model licenses](https://img.shields.io/badge/Model%20licenses-MIT%20%2B%20Apache--2.0-informational.svg)](MODEL_LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.10.2-5A5A5A)](https://github.com/jax-ml/jax)
[![Keras](https://img.shields.io/badge/Keras-3.15.0-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-TPU%20v5e--8-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

A reproducible **JAX / Keras 3 runtime and TPU production engineering project for Mage-Flow-Turbo**, qualified on Kaggle TPU v5e-8.

This repository contains the runtime source, TPU orchestration, correctness contracts, CPU regression tests, acceptance metadata, and documentation used to run the converted Mage-Flow-Turbo model on an 8-device TPU v5e-8 environment. It is an independent conversion/runtime engineering project; it does **not** claim to retrain or replace the upstream Mage-Flow model.

![Mage-Flow-Turbo JAX/Orbax 1024px generation](assets/announcement/hero-1024.png)

## Status

| Item | Qualified value |
| --- | --- |
| Release | `v1.0.0` |
| Qualified runtime commit | `2bf480691789e5db0f49b6c9ab8c134508bff38f` |
| Accelerator | Kaggle TPU v5e-8 |
| TPU devices | 8 |
| Preferred topology | `4x2` replica/model |
| Denoising steps | 4 |
| Qualified resolutions | 512, 768, 1024 |
| Default acceptance seeds | 42, 43, 44, 45 |
| Transformer parameters | 397 leaves |
| Sharding | 174 sharded / 223 replicated |
| VAE runtime binding | 728 / 728 |
| Production acceptance | PASS |

The tagged v1.0.0 runtime is the qualified source baseline. Documentation-only changes on `main`, when present, do not change the accepted inference core unless explicitly stated.

## What this repository provides

- JAX/Keras Transformer runtime source under `runtime/final/source/mage_flow_keras/`.
- A production TPU runner with isolated Text Encoder, Transformer, and VAE stages.
- Explicit topology, resolution, attention, checkpoint, RoPE, and runtime-path controls.
- Fail-closed validation for unsupported attention arguments and unequal packed request lengths.
- Qualified attention policies for 512, 768, and 1024.
- CPU regression tests for attention equivalence, topology/policy contracts, portable paths, and BF16 timestep semantics.
- Machine-readable production acceptance metadata.
- English and Vietnamese documentation for model conversion, TPU execution, benchmarks, reproducibility, limitations, and troubleshooting.
- A public JAX/Orbax model distribution on Hugging Face and Kaggle.

## Public model distribution

This GitHub repository is the **runtime engineering source**. Large converted model artifacts are distributed separately.

| Surface | Purpose |
| --- | --- |
| [Hugging Face model](https://huggingface.co/dangkhoa2016/Mage-Flow-Turbo-JAX-TPU-v5e8) | Primary platform-neutral JAX/Orbax model artifact |
| [Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8) | Kaggle-native model distribution |
| [Kaggle TPU demo](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8-demo) | Ready-to-run TPU demonstration |
| [Upstream Microsoft Mage](https://github.com/microsoft/Mage) | Original model family and research project |

The converted distribution contains native Orbax checkpoints for the Text Encoder, Transformer, and VAE plus tokenizer/configuration and portable runtime support assets.

## Generated examples

The following four 1024×1024 images were generated with the public JAX/Orbax artifact on TPU v5e-8 and manually selected from seed-controlled outputs.

![Selected Mage-Flow-Turbo generations](assets/announcement/sample-grid-4.png)

These images are qualitative examples, **not** benchmark measurements.

## Start here

| Goal | Start with |
| --- | --- |
| Understand what was converted | [Model and conversion](docs/model-and-conversion.md) |
| Run on Kaggle TPU v5e-8 | [Kaggle TPU guide](docs/kaggle-tpu-v5e8.md) |
| Understand the runtime design | [Architecture](docs/architecture.md) |
| Use the production runner | [Inference guide](docs/inference-guide.md) |
| Interpret performance numbers | [Benchmarks](docs/benchmarks.md) |
| Reproduce the qualified result | [Reproducibility](docs/reproducibility.md) |
| Audit release evidence | [Release and verification](docs/release-and-verification.md) |
| Diagnose a failure | [Troubleshooting](docs/troubleshooting.md) |
| Understand boundaries | [Limitations](docs/limitations.md) |
| Browse all documentation | [Documentation Hub](docs/index.md) |
