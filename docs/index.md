# Documentation Hub

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](index.vi.md)

This hub separates public orientation, runtime operation, benchmark interpretation, and release evidence so readers can go directly to the level of detail they need.

## Recommended reading paths

### Run the model
1. [Model and conversion](model-and-conversion.md)
2. [Kaggle TPU v5e-8](kaggle-tpu-v5e8.md)
3. [Inference guide](inference-guide.md)
4. [Troubleshooting](troubleshooting.md)

### Audit the engineering
1. [Architecture](architecture.md)
2. [TPU production runtime](tpu-v5e8-production.md)
3. [Reproducibility](reproducibility.md)
4. [Release and verification](release-and-verification.md)

### Interpret performance
1. [Benchmarks](benchmarks.md)
2. [Limitations](limitations.md)
3. [Production acceptance metadata](../acceptance/PRODUCTION_ACCEPTANCE.json)

## Canonical public surfaces
- Runtime engineering source: this GitHub repository.
- JAX/Orbax model: [Hugging Face](https://huggingface.co/dangkhoa2016/Mage-Flow-Turbo-JAX-TPU-v5e8).
- Kaggle-native model: [Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8).
- Ready-to-run execution: [Kaggle TPU demo](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8-demo).

## Qualification baseline
The production authority records Kaggle TPU v5e-8, 8 devices, topology `4x2`, four denoising steps, seeds 42–45, three qualified resolutions, and deterministic acceptance.
