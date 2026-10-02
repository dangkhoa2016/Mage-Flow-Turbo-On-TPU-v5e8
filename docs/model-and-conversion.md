# Model and Conversion

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](model-and-conversion.vi.md)

## Project boundary
This repository is an independent JAX/Keras runtime and TPU engineering project around upstream Mage-Flow-Turbo. It does not claim that a new foundation model was trained here.

The upstream family is Microsoft Mage/Mage-Flow. The converted public artifact is distributed as `dangkhoa2016/Mage-Flow-Turbo-JAX-TPU-v5e8`.

## Portable conversion target
```text
Text Encoder  -> native Orbax checkpoint
Transformer   -> native Orbax checkpoint
VAE           -> native Orbax checkpoint
Tokenizer     -> config/tokenizer assets
Runtime       -> JAX/Keras source + TPU orchestration
```

Once the converted package is available, the public artifact is intended to infer without the original PyTorch/safetensors weights.

## Qualified engineering contract
- Transformer leaves: 397.
- Sharded: 174.
- Replicated: 223.
- VAE binding: 728/728.
- TPU devices: 8.
- Preferred topology: `4x2`.
- Denoising steps: 4.
- Resolutions: 512, 768, 1024.

The 1024 path uses segmented-query-chunk attention with query chunk 256; 512/768 use segmented attention.

## Text encoder lineage
The Mage-Flow inference stack uses text-encoder components with Qwen3-VL lineage. Those components retain their upstream terms and attribution.

## Distribution responsibilities
- GitHub: runtime source, tests, orchestration, docs, acceptance metadata.
- Hugging Face: platform-neutral converted JAX/Orbax artifact.
- Kaggle Model: Kaggle-native model distribution.
- Kaggle demo: public execution surface.

Qualified runtime source commit: `b9d39a55d8969fc22b9666fa4a415c8523ee3e8f`.
