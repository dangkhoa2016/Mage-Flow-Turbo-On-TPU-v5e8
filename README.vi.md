# Mage-Flow-Turbo trên TPU v5e-8

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8?display_name=tag&sort=semver)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-On-TPU-v5e8/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.10.2-5A5A5A)](https://github.com/jax-ml/jax)
[![Keras](https://img.shields.io/badge/Keras-3.15.0-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-TPU%20v5e--8-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

Dự án **runtime JAX / Keras 3 và kỹ thuật production TPU cho Mage-Flow-Turbo**, đã được qualification trên Kaggle TPU v5e-8 theo hướng có thể tái tạo.

Repository này chứa mã nguồn runtime, orchestration TPU, correctness contract, CPU regression test, acceptance metadata và tài liệu dùng để chạy model Mage-Flow-Turbo đã chuyển đổi trên môi trường TPU v5e-8 gồm 8 thiết bị. Đây là dự án kỹ thuật conversion/runtime độc lập; dự án **không** tuyên bố huấn luyện lại hoặc thay thế model Mage-Flow upstream.

![Ảnh 1024px từ Mage-Flow-Turbo JAX/Orbax](assets/announcement/hero-1024.png)

## Trạng thái

| Hạng mục | Giá trị đã qualify |
| --- | --- |
| Release | `v1.0.0` |
| Qualified runtime commit | `b9d39a55d8969fc22b9666fa4a415c8523ee3e8f` |
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

Runtime gắn tag v1.0.0 là baseline source đã được qualification. Những thay đổi chỉ liên quan đến tài liệu trên `main`, nếu có, không làm thay đổi inference core đã được acceptance trừ khi được ghi rõ.

## Repository này cung cấp gì

- Mã nguồn Transformer JAX/Keras tại `runtime/final/source/mage_flow_keras/`.
- Production TPU runner với các stage Text Encoder, Transformer và VAE chạy trong subprocess tách biệt.
- Điều khiển rõ ràng topology, resolution, attention, checkpoint, RoPE và runtime path.
- Validation fail-closed cho attention argument không hỗ trợ và packed request có chiều dài không đồng đều.
- Attention policy đã qualify cho 512, 768 và 1024.
- CPU regression test cho attention equivalence, topology/policy contract, portable path và BF16 timestep semantics.
- Production acceptance metadata ở dạng machine-readable.
- Bộ tài liệu tiếng Anh và tiếng Việt về conversion, thực thi TPU, benchmark, reproducibility, limitation và troubleshooting.
- Model JAX/Orbax public trên Hugging Face và Kaggle.

## Phân phối model public

GitHub repository này là **runtime engineering source**. Các model artifact lớn được phân phối riêng.

| Nền tảng | Mục đích |
| --- | --- |
| [Hugging Face model](https://huggingface.co/dangkhoa2016/Mage-Flow-Turbo-JAX-TPU-v5e8) | Model artifact JAX/Orbax trung lập nền tảng |
| [Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8) | Phân phối model native cho Kaggle |
| [Kaggle TPU demo](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8-demo) | Demo TPU có thể chạy trực tiếp |
| [Microsoft Mage upstream](https://github.com/microsoft/Mage) | Dự án nghiên cứu và model family gốc |

## Ví dụ ảnh được tạo

Bốn ảnh 1024×1024 bên dưới được tạo bằng public JAX/Orbax artifact trên TPU v5e-8 và được chọn thủ công từ các output điều khiển bằng seed.

![Bốn ảnh Mage-Flow-Turbo được chọn](assets/announcement/sample-grid-4.png)

Đây là ví dụ định tính, **không phải** benchmark.

## Bắt đầu từ đâu

| Mục tiêu | Tài liệu nên đọc |
| --- | --- |
| Hiểu model đã được chuyển đổi như thế nào | [Model và conversion](docs/model-and-conversion.vi.md) |
| Chạy trên Kaggle TPU v5e-8 | [Hướng dẫn Kaggle TPU](docs/kaggle-tpu-v5e8.vi.md) |
| Hiểu thiết kế runtime | [Kiến trúc](docs/architecture.vi.md) |
| Dùng production runner | [Hướng dẫn inference](docs/inference-guide.vi.md) |
| Hiểu đúng số liệu performance | [Benchmark](docs/benchmarks.vi.md) |
| Tái tạo kết quả đã qualify | [Reproducibility](docs/reproducibility.vi.md) |
| Audit release evidence | [Release và verification](docs/release-and-verification.vi.md) |
| Xử lý lỗi | [Troubleshooting](docs/troubleshooting.vi.md) |
| Hiểu các giới hạn | [Limitations](docs/limitations.vi.md) |
| Xem toàn bộ tài liệu | [Documentation Hub](docs/index.vi.md) |

## Profile TPU đã qualify

| Độ phân giải | Attention production | Query chunk |
| --- | --- | ---: |
| 512 | segmented | — |
| 768 | segmented | — |
| 1024 | segmented-query-chunk | 256 |

Production runner nhận topology `1x8`, `2x4`, `4x2`, nhưng production qualification đã công bố dùng **4x2**. Runner đã qualify chỉ chấp nhận lịch denoise bốn bước.

### Performance stage warm

| Độ phân giải | Warm Transformer batch / 4 | Hiệu dụng s/ảnh | Ảnh/phút | Peak HBM/chip | Warm VAE batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 13.124 s | 3.281 | 18.29 | ~9.18 GB | 0.736 s |
| 768 | 15.228 s | 3.807 | 15.76 | ~14.76 GB | 0.750 s |
| 1024 | 33.542 s | 8.385 | 7.16 | ~12.53 GB | 0.782 s |

> **Phạm vi benchmark:** đây là số đo warm của **Transformer stage** và **VAE stage** từ production runner đã qualify. Chúng không phải full cold-start hoặc end-to-end image-generation latency và không nên so sánh trực tiếp với timing GPU full-pipeline.

## Correctness và acceptance

Production acceptance xác minh 8 TPU device, 397 Transformer leaves, 174 sharded / 223 replicated parameters, VAE binding `728/728`, deterministic bit-exact warm rerun, output khác nhau theo seed và PNG byte-identical với qualification output đã review trực quan trước đó.

Authority machine-readable là [`acceptance/PRODUCTION_ACCEPTANCE.json`](acceptance/PRODUCTION_ACCEPTANCE.json).

## Quick start

Repository chủ động không lưu checkpoint Orbax lớn hoặc toàn bộ pinned runtime cache. Hãy attach/download public model artifact trước, sau đó truyền các đường dẫn model và runtime cho runner.

```bash
python3 bootstrap/06_run_tpu_inference.py \
  --topology 4x2 --resolution 1024 --seeds 42,43,44,45 --steps 4 \
  --attention auto --model-root "$MODEL_ROOT" --runtime-root "$RUNTIME_ROOT" \
  --runtime-site "$RUNTIME_SITE" --text-checkpoint "$TEXT_CHECKPOINT" \
  --transformer-checkpoint "$TRANSFORMER_CHECKPOINT" --vae-checkpoint "$VAE_CHECKPOINT" \
  --vae-manifest "$VAE_MANIFEST" --basis-dim16 "$BASIS_DIM16" \
  --basis-dim56 "$BASIS_DIM56" --output "$OUTPUT_DIR"
```

## Development và CI

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q runtime bootstrap tests
pytest -q tests/test_cpu_regressions.py
python scripts/check_docs.py
git diff --check
```

CPU CI kiểm tra contract ở cấp source; nó **không** thay thế qualification run thật trên TPU.

## Tài liệu

README là landing page. Tài liệu chi tiết nằm trong [Documentation Hub](docs/index.vi.md).

| Chủ đề | English | Tiếng Việt |
| --- | --- | --- |
| Architecture | [Open](docs/architecture.md) | [Mở](docs/architecture.vi.md) |
| Model và conversion | [Open](docs/model-and-conversion.md) | [Mở](docs/model-and-conversion.vi.md) |
| Kaggle TPU v5e-8 | [Open](docs/kaggle-tpu-v5e8.md) | [Mở](docs/kaggle-tpu-v5e8.vi.md) |
| Inference guide | [Open](docs/inference-guide.md) | [Mở](docs/inference-guide.vi.md) |
| Benchmark | [Open](docs/benchmarks.md) | [Mở](docs/benchmarks.vi.md) |
| Reproducibility | [Open](docs/reproducibility.md) | [Mở](docs/reproducibility.vi.md) |
| Release và verification | [Open](docs/release-and-verification.md) | [Mở](docs/release-and-verification.vi.md) |
| Limitations | [Open](docs/limitations.md) | [Mở](docs/limitations.vi.md) |
| Troubleshooting | [Open](docs/troubleshooting.md) | [Mở](docs/troubleshooting.vi.md) |
| Production runtime authority | [Open](docs/tpu-v5e8-production.md) | [Mở](docs/tpu-v5e8-production.vi.md) |

## License và upstream attribution

Mã nguồn kỹ thuật và tài liệu gốc của repository này được phát hành theo [MIT License](LICENSE). Các thành phần model Mage-Flow và Text Encoder có lineage Qwen3-VL giữ nguyên license/attribution upstream. Repository này không relicensing model weights hoặc dependency bên thứ ba.

Xem [NOTICE.md](NOTICE.md) và [Model và conversion](docs/model-and-conversion.vi.md).

## Tác giả

**Đăng Khoa**

`i.am@dangkhoa.dev`

## Lời cảm ơn

Dự án được xây dựng dựa trên Microsoft Mage/Mage-Flow, JAX, Keras, Orbax, các thành phần Qwen3-VL, hạ tầng Kaggle TPU và hệ sinh thái Python mã nguồn mở. Các dự án, thương hiệu, model, dịch vụ và license của họ vẫn độc lập với repository này.
