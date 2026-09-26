# Changelog

## v1.0.0 — 2026-10-02

First public release. TPU runtime qualification was completed on 2026-09-25 at commit `2bf480691789e5db0f49b6c9ab8c134508bff38f`.

- Added the portable TPU production runner.
- Added qualified JAX/Keras Mage-Flow source.
- Added CPU regression coverage.
- Recorded TPU v5e-8 production profile and acceptance metadata.
- Qualified 512, 768, and 1024 resolutions on the documented 4x2 production profile.

## Release documentation and governance

The v1.0.0 release includes README presentation, bilingual navigation, public model links, documentation structure, showcase media, licensing metadata, and repository governance without changing the qualified inference core.

### Repository governance and licensing hardening

- Added community-health files under `.github/`.
- Added explicit model-component licensing boundaries in English and Vietnamese.
- Preserved upstream Mage-Flow MIT and Qwen3-VL Apache-2.0 license texts.
- Added third-party notices and repository-health CI validation.
- Kept the repository-level MIT `LICENSE` unchanged.
