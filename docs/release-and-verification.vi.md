# Release và Verification

> 🌐 Ngôn ngữ / Language: [English](release-and-verification.md) | **Tiếng Việt**

## Source authority v1.0.0
Qualified runtime commit:
```text
2bf480691789e5db0f49b6c9ab8c134508bff38f
```

Repository ghi nhận production acceptance ngày 2026-09-25. Commit `2bf480691789e5db0f49b6c9ab8c134508bff38f` là authority bất biến của runtime qualification. Tag `v1.0.0` xác định v1.0.0 public release, đóng gói runtime đã qualify cùng tài liệu release, licensing, governance và showcase.

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

Thay đổi runtime semantics cần được chạy lại trên TPU v5e-8 theo qualified profile và so với commit `2bf480691789e5db0f49b6c9ab8c134508bff38f` cùng accepted evidence trước khi mô tả là tương đương với runtime đã release.

## Ranh giới release
Documentation, badge, navigation, licensing metadata, governance file và public showcase media có thể nằm trong v1.0.0 release mà không thay đổi qualified inference core. Runtime-semantic modification phải được tách rõ và cần TPU qualification mới.
