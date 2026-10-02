# Documentation Hub

> 🌐 Ngôn ngữ / Language: [English](index.md) | **Tiếng Việt**

Hub này tách riêng phần giới thiệu public, vận hành runtime, diễn giải benchmark và release evidence để người đọc đi thẳng tới mức chi tiết cần thiết.

## Lộ trình đọc đề xuất

### Chạy model
1. [Model và conversion](model-and-conversion.vi.md)
2. [Kaggle TPU v5e-8](kaggle-tpu-v5e8.vi.md)
3. [Hướng dẫn inference](inference-guide.vi.md)
4. [Troubleshooting](troubleshooting.vi.md)

### Audit engineering
1. [Kiến trúc](architecture.vi.md)
2. [TPU production runtime](tpu-v5e8-production.vi.md)
3. [Reproducibility](reproducibility.vi.md)
4. [Release và verification](release-and-verification.vi.md)

### Hiểu performance
1. [Benchmark](benchmarks.vi.md)
2. [Limitations](limitations.vi.md)
3. [Production acceptance metadata](../acceptance/PRODUCTION_ACCEPTANCE.json)

## Public surface chuẩn
- Runtime engineering source: GitHub repository này.
- JAX/Orbax model: [Hugging Face](https://huggingface.co/dangkhoa2016/Mage-Flow-Turbo-JAX-TPU-v5e8).
- Kaggle-native model: [Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8).
- Execution có thể chạy trực tiếp: [Kaggle TPU demo](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8-demo).

## Qualification baseline
Production authority ghi nhận Kaggle TPU v5e-8, 8 thiết bị, topology `4x2`, bốn bước denoise, seed 42–45, ba độ phân giải đã qualify và deterministic acceptance.
