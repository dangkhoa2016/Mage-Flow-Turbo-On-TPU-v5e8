# Release và Verification

> 🌐 Ngôn ngữ / Language: [English](release-and-verification.md) | **Tiếng Việt**

## Source authority v1.0.0
Qualified runtime commit:
```text
2bf480691789e5db0f49b6c9ab8c134508bff38f
```

Repository ghi nhận production acceptance ngày 2026-09-25. Tag `v1.0.0` xác định baseline release đã qualify.

## Acceptance artifacts
- [`PRODUCTION_ACCEPTANCE.json`](../acceptance/PRODUCTION_ACCEPTANCE.json) — policy/result machine-readable.
- [`PRODUCTION_ACCEPTANCE_CLOSEOUT.md`](../acceptance/PRODUCTION_ACCEPTANCE_CLOSEOUT.md) — closeout dạng dễ đọc.
- [`tpu-v5e8-production.vi.md`](tpu-v5e8-production.vi.md) — production runtime contract.

Evidence archive SHA-256:
```text
4982c750914599561cd5255c1ad9e58cfb02a9f2c26a299801271a92d2c0bf68
```

## CI và TPU verification
GitHub CI compile Python source, chạy CPU regression test và validate documentation. CI không thể chứng nhận TPU hardware behavior.

Thay đổi runtime semantics cần được chạy lại trên TPU v5e-8 theo qualified profile và so với accepted authority trước khi mô tả là tương đương v1.0.0.

## Documentation-only changes
Documentation, badge, navigation và public showcase media có thể phát triển trên `main` mà không thay đổi inference core v1.0.0. Các thay đổi này cần được tách rõ khỏi runtime-semantic modification.
