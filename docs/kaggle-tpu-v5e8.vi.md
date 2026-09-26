# Hướng dẫn Kaggle TPU v5e-8

> 🌐 Ngôn ngữ / Language: [English](kaggle-tpu-v5e8.md) | **Tiếng Việt**

## Môi trường đã qualify
Production qualification chính thức nhắm tới Kaggle TPU v5e-8 với 8 TPU device. Profile đã công bố dùng topology `4x2`, bốn bước denoise và seed 42–45 cho acceptance.

## Public path được khuyến nghị
Để tái tạo nhanh nhất, dùng [Kaggle TPU demo](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8-demo) public và attach [Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/mage-flow-turbo-jax-tpu-v5e8).

GitHub runner chủ yếu là bề mặt source/audit. Runner yêu cầu đường dẫn rõ ràng tới model checkpoints, runtime support, VAE manifest và RoPE basis assets.

## Kỳ vọng preflight
Trước inference cần xác minh:
- JAX nhìn thấy TPU backend;
- có đúng 8 TPU device;
- topology yêu cầu tương thích với 8 thiết bị;
- các model/runtime path bắt buộc tồn tại;
- resolution là 512, 768 hoặc 1024;
- `--steps` đúng bằng 4 với qualified runner.

Session fallback im lặng sang CPU không được coi là TPU reproduction thành công.

## Production policy
Dùng `--attention auto` trừ khi chủ động tái tạo diagnostic cấp thấp. Auto chọn attention mode đã qualify theo resolution.

CPU CI hữu ích trước TPU run, nhưng không chứng minh TPU performance hoặc end-to-end acceptance.
