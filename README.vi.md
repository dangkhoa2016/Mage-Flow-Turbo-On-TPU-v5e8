# Mage-Flow Turbo trên TPU v5e-8

Mã nguồn runtime Keras 3 / JAX và lớp orchestration production cho Mage-Flow Turbo trên Kaggle TPU v5e-8.

Profile production đã qualify sử dụng tám TPU device với topology **4x2 replica/model** ưu tiên, bốn bước denoise, segmented attention ở 512/768 và segmented query-chunk attention ở 1024.

## Profile production

| Độ phân giải | Attention | Query chunk | Warm transformer batch | Ảnh/phút |
|---|---|---:|---:|---:|
| 512 | segmented | — | 13.124 s / 4 | 18.29 |
| 768 | segmented | — | 15.228 s / 4 | 15.76 |
| 1024 | segmented-query-chunk | 256 | 33.542 s / 4 | 7.16 |

Acceptance cũng xác minh 397 transformer leaves, 174 sharded / 223 replicated parameters, VAE binding 728/728, warm rerun bit-exact, output khác nhau theo seed và PNG byte-identical với các qualification run đã được review trực quan.

## Cấu trúc repository

```text
runtime/final/source/mage_flow_keras/  mã nguồn transformer JAX/Keras đã qualify
bootstrap/                             production TPU runner và stage helpers
tests/                                 CPU regression tests
docs/                                  tài liệu production EN/VI
acceptance/                            metadata acceptance gọn nhẹ
```

## Runtime prerequisites

Các checkpoint lớn, pinned Python runtime cache và package runtime support `experiment` bên ngoài được chủ động không lưu trong Git repository này.

Production runner nhận các đường dẫn đó qua `--runtime-root`, `--runtime-site` tùy chọn, checkpoint arguments, RoPE basis inputs và canonical model root. Cách này giữ repository dễ review nhưng vẫn bảo toàn các runtime contract có thể tái tạo.

Xem [tài liệu production TPU v5e-8](docs/tpu-v5e8-production.vi.md) để biết policy đã qualify, benchmark, correctness constraints và yêu cầu verification trước release.
