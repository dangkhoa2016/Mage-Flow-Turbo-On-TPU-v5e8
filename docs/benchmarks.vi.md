# Benchmark

> 🌐 Ngôn ngữ / Language: [English](benchmarks.md) | **Tiếng Việt**

## Số đo đã qualify
| Độ phân giải | Warm Transformer batch / 4 | Hiệu dụng s/ảnh | Ảnh/phút | Peak HBM/chip | Warm VAE batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 13.124 s | 3.281 | 18.29 | ~9.18 GB | 0.736 s |
| 768 | 15.228 s | 3.807 | 15.76 | ~14.76 GB | 0.750 s |
| 1024 | 33.542 s | 8.385 | 7.16 | ~12.53 GB | 0.782 s |

Nguồn: [`acceptance/PRODUCTION_ACCEPTANCE.json`](../acceptance/PRODUCTION_ACCEPTANCE.json).

## Các số này có nghĩa gì
Số Transformer là **warm Transformer-stage measurement** cho batch bốn ảnh. s/ảnh hiệu dụng và ảnh/phút được suy ra từ stage batch time này. Số VAE là warm VAE-stage batch measurement.

## Các số này không có nghĩa gì
Chúng không phải:
- full cold-start latency;
- thời gian download model hoặc cài runtime;
- wall time end-to-end Text Encoder + Transformer + VAE;
- số có thể so trực tiếp với GPU benchmark có timing scope khác.

## Cách so sánh đúng
So sánh công bằng nên khớp prompt, seed, resolution, denoising steps, timing boundary, warm/cold state và batch semantics. Nếu không khớp, hãy gọi đó là reference thay vì tuyên bố speedup apples-to-apples.
