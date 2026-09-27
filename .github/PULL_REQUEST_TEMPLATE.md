## Summary

Describe the change and why it is needed.

## Scope

- [ ] Documentation / metadata only
- [ ] CI / tests only
- [ ] Runtime behavior
- [ ] TPU semantics / sharding / attention
- [ ] Licensing / attribution

## Validation

- [ ] `python -m compileall -q runtime bootstrap tests scripts`
- [ ] `KERAS_BACKEND=jax pytest -q tests/test_cpu_regressions.py`
- [ ] `python scripts/check_docs.py`
- [ ] `python scripts/check_repo_health.py`
- [ ] `git diff --check`

## TPU qualification

- [ ] Not required; no TPU semantics changed
- [ ] Required and completed
- [ ] Required but not yet completed

Provide TPU evidence when applicable.

## Licensing and attribution

- [ ] No upstream licensing/attribution removed
- [ ] New third-party material is documented

## Checklist

- [ ] No credentials or private data included
- [ ] EN/VI documentation kept in sync where applicable
- [ ] Change is focused and reviewable
