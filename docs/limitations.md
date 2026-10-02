# Limitations

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](limitations.vi.md)

## Qualification boundary
The formal qualification covers the documented Kaggle TPU v5e-8 production profile. It does not establish equivalent behavior on every TPU generation, GPU, CPU, cloud provider, or arbitrary dependency stack.

## Benchmark boundary
Published performance numbers are warm stage-level measurements, not complete application latency.

## Model behavior
Image quality and prompt adherence vary by prompt. This conversion/runtime project preserves upstream model behavior rather than creating a new training objective.

## Safety and data scope
This project does not include a new training-data audit, demographic-fairness benchmark, or application-specific safety certification.

## Dependency sensitivity
Changing JAX, Keras, Orbax behavior, sharding rules, attention implementation, BF16 timestep semantics, or denoising schedule can invalidate qualification.

## Operational scope
The repository is an inference engineering project, not an SLA-backed hosted service. Multi-tenancy, autoscaling, abuse prevention, and availability are outside this qualification.
