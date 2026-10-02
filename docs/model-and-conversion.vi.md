# Model và Conversion

> 🌐 Ngôn ngữ / Language: [English](model-and-conversion.md) | **Tiếng Việt**

## Ranh giới dự án
Repository này là dự án runtime JAX/Keras và kỹ thuật TPU độc lập cho Mage-Flow-Turbo upstream. Dự án không tuyên bố huấn luyện một foundation model mới.

Model family upstream là Microsoft Mage/Mage-Flow. Public artifact đã chuyển đổi được phân phối với tên `dangkhoa2016/Mage-Flow-Turbo-JAX-TPU-v5e8`.

## Mục tiêu conversion portable
```text
Text Encoder  -> native Orbax checkpoint
Transformer   -> native Orbax checkpoint
VAE           -> native Orbax checkpoint
Tokenizer     -> config/tokenizer assets
Runtime       -> JAX/Keras source + TPU orchestration
```

Khi package conversion đã có sẵn, public artifact được thiết kế để inference mà không cần original PyTorch/safetensors weights.

## Engineering contract đã qualify
- Transformer leaves: 397.
- Sharded: 174.
- Replicated: 223.
- VAE binding: 728/728.
- TPU devices: 8.
- Topology ưu tiên: `4x2`.
- Denoising steps: 4.
- Độ phân giải: 512, 768, 1024.

Đường 1024 dùng segmented-query-chunk attention với query chunk 256; 512/768 dùng segmented attention.

## Lineage Text Encoder
Inference stack Mage-Flow dùng các thành phần Text Encoder có lineage Qwen3-VL. Các thành phần đó giữ nguyên điều khoản và attribution upstream.

## Trách nhiệm của từng bề mặt
- GitHub: runtime source, tests, orchestration, docs, acceptance metadata.
- Hugging Face: converted JAX/Orbax artifact trung lập nền tảng.
- Kaggle Model: phân phối model native cho Kaggle.
- Kaggle demo: public execution surface.

Qualified runtime source commit: `b9d39a55d8969fc22b9666fa4a415c8523ee3e8f`.
