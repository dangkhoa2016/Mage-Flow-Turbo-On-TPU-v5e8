# Mage-Flow TPU v5e-8 Production Acceptance — 2026-09-25

Status: PASS

- Topology: 4x2
- Steps: 4
- Seeds: 42,43,44,45
- Policy: 512=segmented; 768=segmented; 1024=segmented-query-chunk(256)

| Resolution | Attention | Warm transformer batch | Effective s/image | Images/min | Peak HBM/chip | Warm VAE batch |
|---|---|---:|---:|---:|---:|---:|
| 512 | segmented | 13.124s | 3.281s | 18.29 | 9.18 GB | 0.736s |
| 768 | segmented | 15.228s | 3.807s | 15.76 | 14.76 GB | 0.750s |
| 1024 | segmented-query-chunk (256) | 33.542s | 8.385s | 7.16 | 12.53 GB | 0.782s |

## Acceptance

- Transformer deterministic warm rerun: PASS
- Per-seed outputs unique: PASS
- VAE binding coverage 728/728: PASS
- PNG outputs byte-identical to the previously visually accepted qualification runs at 512, 768, and 1024: PASS
- Production source no longer requires runtime monkey-patching for segmented/query-chunk attention: PASS

## Evidence contents

- summaries/: production and stage JSON summaries
- logs/: per-stage logs
- png/: 12 accepted PNG outputs
- source/: promoted attention source and full transformer source
- bootstrap/: production runner and isolated stage helpers
- SHA256SUMS.txt: file hashes
