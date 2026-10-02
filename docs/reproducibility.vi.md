# Reproducibility

> 🌐 Ngôn ngữ / Language: [English](reproducibility.md) | **Tiếng Việt**

## Baseline đã acceptance
```text
accelerator   Kaggle TPU v5e-8
devices       8
topology      4x2
steps         4
seeds         42,43,44,45
resolutions   512,768,1024
```

## Authority chain
1. Runtime source được pin bởi qualified commit v1.0.0.
2. Converted model artifact được cung cấp qua public model distribution.
3. Production runner resolve attention policy đã qualify.
4. Text Encoder, Transformer và VAE chạy thành các stage tách biệt.
5. Acceptance kiểm tra deterministic warm rerun và output khác nhau theo seed.
6. PNG cuối được so với qualification output đã review trực quan trước đó.

## Authority machine-readable
[`acceptance/PRODUCTION_ACCEPTANCE.json`](../acceptance/PRODUCTION_ACCEPTANCE.json) chứa policy, timing, HBM và SHA-256 PNG đã acceptance theo từng resolution.

## Điều gì làm mất tính tương đương
Hãy coi run là experiment mới nếu thay đổi JAX/Keras version, Transformer source, BF16 timestep semantics, sharding/topology, attention policy, denoising steps, checkpoint mapping, VAE binding hoặc resolution ngoài tập đã qualify.

CPU test có thể bắt nhiều source-contract regression, nhưng TPU qualification vẫn là hardware authority.
