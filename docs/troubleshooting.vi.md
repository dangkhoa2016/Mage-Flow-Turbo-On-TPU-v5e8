# Troubleshooting

> 🌐 Ngôn ngữ / Language: [English](troubleshooting.md) | **Tiếng Việt**

## JAX không thấy TPU / device count không phải 8
Không tiếp tục như thể run đã qualify. Trước tiên kiểm tra accelerator setting của Kaggle và TPU runtime environment.

## Runner từ chối `--steps`
Qualified runner chỉ nhận bốn bước denoise. Đây là fail-closed behavior có chủ đích.

## Attention mode hoặc kwargs không hỗ trợ
Dùng `--attention auto` cho production policy. Unknown mode, unsupported joint attention kwargs, query chunk không hợp lệ hoặc packed request có chiều dài không đồng đều bị từ chối theo thiết kế.

## Thiếu checkpoint/runtime path
Git repository không chứa checkpoint model lớn hoặc toàn bộ runtime cache. Hãy attach/download public model artifact và truyền đường dẫn rõ ràng.

## CPU test PASS nhưng TPU run lỗi
CPU regression test kiểm tra source contract; chúng không thay thế TPU runtime initialization, sharding, HBM availability hoặc device execution.

## Performance khác README
Xác minh đang so cùng timing boundary. Số README là warm Transformer/VAE stage measurement, không phải cold end-to-end latency.

## Output khác accepted hash
Kiểm tra source commit, model artifact identity, seed, resolution, steps, attention policy, runtime version, sharding, RoPE assets và VAE manifest trước khi coi khác biệt là model nondeterminism.
