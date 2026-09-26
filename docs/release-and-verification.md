# Release and Verification

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](release-and-verification.vi.md)

## v1.0.0 source authority
Qualified runtime commit:
```text
2bf480691789e5db0f49b6c9ab8c134508bff38f
```

The repository records production acceptance on 2026-09-25. The `v1.0.0` tag identifies the qualified release baseline.

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

A runtime-semantic change should be rerun on TPU v5e-8 at the qualified profile and compared with accepted authority before being described as equivalent to v1.0.0.

## Documentation-only changes
Documentation, badges, navigation, and public showcase media may evolve on `main` without changing the v1.0.0 inference core. Such changes should remain clearly separated from runtime-semantic modifications.
