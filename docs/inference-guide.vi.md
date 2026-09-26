# Hướng dẫn Inference

> 🌐 Ngôn ngữ / Language: [English](inference-guide.md) | **Tiếng Việt**

Production entry point là `bootstrap/06_run_tpu_inference.py`.

## Argument chính
| Argument | Contract |
| --- | --- |
| `--topology` | `1x8`, `2x4` hoặc `4x2`; qualification dùng `4x2` |
| `--resolution` | 512, 768 hoặc 1024 |
| `--seeds` | danh sách seed phân cách bằng dấu phẩy |
| `--steps` | qualified runner yêu cầu 4 |
| `--attention` | `auto`, `masked-global`, `segmented`, `segmented-query-chunk` |
| `--query-chunk` | số nguyên dương; giá trị 1024 đã qualify là 256 |
| `--model-root` | canonical model root |
| `--runtime-root` | runtime support root |
| `--runtime-site` | isolated Python site tùy chọn |
| checkpoint args | checkpoint Text Encoder, Transformer và VAE rõ ràng |
| RoPE args | basis asset dimension-16 và dimension-56 rõ ràng |
| `--output` | thư mục output |

## Fail-closed behavior
Runner từ chối denoising schedule khác bốn bước. Attention runtime từ chối mode không biết, query chunk không dương, attention kwargs không hỗ trợ và packed request có chiều dài không đồng đều trong segmented production mode.

Đây là chủ đích: configuration không được qualify phải fail rõ ràng thay vì âm thầm tạo kết quả không có authority.

## Setting production khuyến nghị
```text
topology    4x2
steps       4
attention   auto
seeds       42,43,44,45 (acceptance baseline)
```

Dùng [Reproducibility](reproducibility.vi.md) khi muốn tái tạo baseline đã acceptance thay vì chỉ thử nghiệm runtime.
