# Troubleshooting

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](troubleshooting.vi.md)

## JAX does not report TPU / device count is not 8
Do not continue as if the run were qualified. Verify the Kaggle accelerator setting and TPU runtime environment first.

## Runner rejects `--steps`
The qualified runner accepts only four denoising steps. This is intentional fail-closed behavior.

## Unsupported attention mode or kwargs
Use `--attention auto` for the production policy. Unknown modes, unsupported joint attention kwargs, invalid query chunks, or unequal packed request lengths are rejected by design.

## Missing checkpoint/runtime path
The Git repository does not contain the large model checkpoints or complete runtime cache. Attach/download the public model artifact and pass explicit paths.

## CPU tests pass but TPU run fails
CPU regression tests validate source contracts; they do not replace TPU runtime initialization, sharding, HBM availability, or device execution.

## Performance differs from the README
Confirm that you are comparing the same timing boundary. README figures are warm Transformer/VAE stage measurements, not cold end-to-end latency.

## Output differs from accepted hashes
Check source commit, model artifact identity, seed, resolution, steps, attention policy, runtime versions, sharding, RoPE assets, and VAE manifest before treating the difference as model nondeterminism.
