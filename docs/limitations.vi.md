# Limitations

> 🌐 Ngôn ngữ / Language: [English](limitations.md) | **Tiếng Việt**

## Ranh giới qualification
Qualification chính thức bao phủ profile production Kaggle TPU v5e-8 đã ghi. Nó không chứng minh behavior tương đương trên mọi thế hệ TPU, GPU, CPU, cloud provider hoặc dependency stack tùy ý.

## Ranh giới benchmark
Performance đã công bố là số đo warm theo stage, không phải complete application latency.

## Behavior của model
Image quality và prompt adherence thay đổi theo prompt. Dự án conversion/runtime này bảo toàn behavior của model upstream thay vì tạo training objective mới.

## Phạm vi safety và data
Dự án không bao gồm training-data audit mới, demographic-fairness benchmark hoặc application-specific safety certification.

## Độ nhạy dependency
Thay đổi JAX, Keras, Orbax behavior, sharding rule, attention implementation, BF16 timestep semantics hoặc denoising schedule có thể làm mất hiệu lực qualification.

## Phạm vi vận hành
Repository là dự án inference engineering, không phải hosted service có SLA. Multi-tenancy, autoscaling, abuse prevention và availability nằm ngoài qualification này.
