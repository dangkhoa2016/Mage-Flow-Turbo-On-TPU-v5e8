# Release and Verification

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](release-and-verification.vi.md)

## v1.0.0 source authority
Qualified runtime commit:
```text
2bf480691789e5db0f49b6c9ab8c134508bff38f
```

The repository records production acceptance on 2026-09-25. Commit `2bf480691789e5db0f49b6c9ab8c134508bff38f` is the immutable runtime qualification authority. The `v1.0.0` tag identifies the v1.0.0 public release that packages this qualified runtime together with release documentation, licensing, governance, and showcase material.

## Acceptance artifacts
- [`PRODUCTION_ACCEPTANCE.json`](../acceptance/PRODUCTION_ACCEPTANCE.json) — machine-readable policy/results.
- [`PRODUCTION_ACCEPTANCE_CLOSEOUT.md`](../acceptance/PRODUCTION_ACCEPTANCE_CLOSEOUT.md) — compact human-readable closeout.
- [`tpu-v5e8-production.md`](tpu-v5e8-production.md) — production runtime contract.

Evidence archive SHA-256:
```text
4982c750914599561cd5255c1ad9e58cfb02a9f2c26a299801271a92d2c0bf68
```

## CI versus TPU verification
GitHub CI compiles Python source, runs CPU regression tests, and validates documentation. It cannot certify TPU hardware behavior.

A runtime-semantic change should be rerun on TPU v5e-8 at the qualified profile and compared with commit `2bf480691789e5db0f49b6c9ab8c134508bff38f` plus the accepted evidence before being described as equivalent to the released runtime.

## Release boundary
Documentation, badges, navigation, licensing metadata, governance files, and public showcase media may be part of the v1.0.0 release without changing the qualified inference core. Runtime-semantic modifications must remain clearly separated and require new TPU qualification.
