import ast
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCK0_PATH = ROOT / "runtime" / "final" / "source" / "mage_flow_keras" / "block0_candidate.py"
RUNNER_PATH = ROOT / "bootstrap" / "06_run_tpu_inference.py"
FULL_MODEL_PATH = ROOT / "runtime" / "final" / "source" / "mage_flow_keras" / "full_model.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


block0 = load_module("block0_candidate_test", BLOCK0_PATH)
runner = load_module("run_tpu_inference_test", RUNNER_PATH)


def tiny_candidate():
    return block0.MageFlowBlock0Candidate(
        dim=4,
        num_attention_heads=1,
        attention_head_dim=4,
    )


def attention_inputs():
    q = jnp.arange(32, dtype=jnp.float32).reshape(8, 1, 4) / 17.0
    k = jnp.flip(q, axis=0)
    v = (q * 0.25) + 0.125
    cu = jnp.asarray([0, 4, 8], dtype=jnp.int32)
    return q, k, v, cu


def test_attention_modes_match_reference():
    model = tiny_candidate()
    q, k, v, cu = attention_inputs()
    masked = model._segment_attention(q, k, v, cu, 2, attention_mode="masked-global")
    segmented = model._segment_attention(q, k, v, cu, 2, attention_mode="segmented")
    chunked = model._segment_attention(
        q, k, v, cu, 2, attention_mode="segmented-query-chunk", query_chunk=2
    )
    # CPU backends may differ by a few float32 ULPs while preserving the same semantics.
    np.testing.assert_allclose(segmented, masked, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunked, masked, rtol=1e-6, atol=1e-6)


def test_attention_invalid_modes_fail_closed():
    model = tiny_candidate()
    q, k, v, cu = attention_inputs()
    with pytest.raises(block0.MageFlowBlock0Error):
        model._segment_attention(q, k, v, cu, 2, attention_mode="unknown")
    with pytest.raises(block0.MageFlowBlock0Error):
        model._segment_attention(
            q, k, v, cu, 2, attention_mode="segmented-query-chunk", query_chunk=0
        )


def test_unequal_segments_fail_closed():
    model = tiny_candidate()
    q = jnp.zeros((8, 1, 4), dtype=jnp.float32)
    with pytest.raises(block0.MageFlowBlock0Error):
        model._segment_attention(
            q, q, q, jnp.asarray([0, 3, 8], dtype=jnp.int32), 2,
            attention_mode="segmented",
        )


def test_runner_policy_and_topologies_are_frozen():
    assert runner.POLICY == {
        512: ("segmented", 256),
        768: ("segmented", 256),
        1024: ("segmented-query-chunk", 256),
    }
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'choices=("1x8","2x4","4x2")' in source
    assert 'if args.steps!=4' in source


def test_bf16_timestep_semantics_are_preserved():
    source = FULL_MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    normalized = ast.unparse(tree)
    assert "jnp.asarray(timesteps, dtype=jnp.bfloat16).astype(jnp.float32)" in normalized


def test_attention_kwargs_fail_closed_contract_present():
    source = BLOCK0_PATH.read_text(encoding="utf-8")
    assert "unsupported joint_attention_kwargs" in source
    assert "equal packed request lengths are required" in source
    assert "query_chunk must be > 0" in source


def test_runner_uses_portable_runtime_paths():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "PROJECT_ROOT = SCRIPT_DIR.parent" in source
    assert "--runtime-root" in source
    assert "--runtime-site" in source
    assert "--source-root" in source

