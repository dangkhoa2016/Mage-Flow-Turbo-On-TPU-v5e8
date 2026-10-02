# Mage-Flow TPU v5e-8 — Runtime Production

> 🌐 Ngôn ngữ / Language: [English](tpu-v5e8-production.md) | **Tiếng Việt**

Tài liệu này ghi lại runtime production đã vượt qua acceptance ngày 2026-09-25.

## Hợp đồng đã qualify

- Accelerator: Kaggle TPU v5e-8, 8 thiết bị.
- Topology ưu tiên: 4x2.
- Số bước denoise: 4.
- Seed mặc định của acceptance 4x2: 42, 43, 44, 45.
- Transformer: 397 parameter leaves; 174 sharded và 223 replicated.
- VAE runtime binding: 728/728.
- Production source: `runtime/final/source/mage_flow_keras`.

## Chính sách attention

| Độ phân giải | Attention production | Query chunk |
|---|---|---:|
| 512 | segmented | không áp dụng |
| 768 | segmented | không áp dụng |
| 1024 | segmented-query-chunk | 256 |

Các mode segmented production cố ý yêu cầu packed request có cùng chiều dài và sẽ fail closed nếu điều kiện này không thỏa.

## Benchmark production chuẩn

| Độ phân giải | Warm transformer batch | Hiệu dụng s/ảnh | Ảnh/phút | Peak HBM/chip | Warm VAE batch |
|---|---:|---:|---:|---:|---:|
| 512 | 13.124 s / 4 | 3.281 | 18.29 | ~9.18 GB | 0.736 s |
| 768 | 15.228 s / 4 | 3.807 | 15.76 | ~14.76 GB | 0.750 s |
| 1024 | 33.542 s / 4 | 8.385 | 7.16 | ~12.53 GB | 0.782 s |

Các số liệu trên đến từ production runner cuối cùng đã promote, không phải script thử nghiệm.

> Đây là số đo warm theo stage. Chúng không phải full cold-start hoặc end-to-end image-generation latency.

## Yêu cầu về tính đúng

Không được làm regress BF16 timestep semantics: `jnp.asarray(timesteps, dtype=jnp.bfloat16).astype(jnp.float32)`.

Phải construct/restore model state trên CPU trước khi `device_put` vào TPU mesh. Giữ per-request segmented attention cho concurrent execution và query chunk 256 ở 1024 trừ khi có TPU qualification mới thay thế authority này.

Output production ở 512, 768 và 1024 đã byte-identical với các PNG qualification đã được review trực quan trước đó.

## Runner

Dùng `bootstrap/06_run_tpu_inference.py` với các subprocess stage tách biệt cho text encoder, transformer và VAE.

Runner hỗ trợ topology `1x8`, `2x4`, `4x2`. Production runner chỉ chấp nhận lịch bốn bước đã qualify.

## Xác minh release

CPU test và CI bao phủ repository integration; chúng không thay thế TPU release verification. Release candidate thay đổi runtime semantics cần được chạy lại trên TPU v5e-8 ở 4x2 cho 512, 768 và 1024 rồi đối chiếu accepted evidence authority.

Evidence archive SHA-256: `4982c750914599561cd5255c1ad9e58cfb02a9f2c26a299801271a92d2c0bf68`.

Xem [Release và verification](release-and-verification.vi.md) và [Benchmark](benchmarks.vi.md).
