# Resource measurement: S. venezuelae ATCC 10712

Measured on 2026-11-06 (Asia/Seoul), one complete preparation/prediction run.

- Genome: CP029197.1, 7,132 retained genes / 4,995 inferred operons.
- Medium: ISP2. ESM batch size: 16. CPU threads: 4.
- GPU: NVIDIA A100 80GB PCIe; PyTorch 2.0.1+cu117.
- NVML and host/process RAM sampled every 0.2 seconds; OS maximum RSS and PyTorch
  allocator high-water marks also recorded inside each worker.
- Model downloads were already cached. OS file caches were not cleared.
- Input preparation was recomputed; prediction loaded the completed model without fitting.

| Stage | Wall time | Peak GPU busy | Peak device memory | Peak process RSS |
|---|---:|---:|---:|---:|
| Input preparation | 46.06 s | 100% | 54.84 GiB | 3.04 GiB |
| Saved-model prediction | 10.21 s | 74% | 1.98 GiB | 11.07 GiB |

Total: **56.27 seconds**, including Python/model loading and process shutdown.
The prediction function itself reported **4.1 seconds**.

The large device-memory peak occurred during ESM extraction. PyTorch's peak
allocated tensor memory was **2.74 GiB**, versus **53.66 GiB** reserved by its
allocator. Most of the device occupancy was cached/reserved memory. After ESM,
`empty_cache()` released it; the sampled gLM-stage peak was **2.39 GiB**.
Prediction's PyTorch peaks were **0.76 GiB allocated / 0.80 GiB reserved**.
NVML includes CUDA context and other driver allocations as well.

Process RSS includes resident pages of the memory-mapped reference index. It is
not the same metric as system-wide used memory or container memory including file cache.
All 7,132 predictions were finite and matched the requested gene IDs.

These results do not establish a minimum GPU/RAM requirement or a latency guarantee.
A smaller-VRAM GPU and different genome sizes have not been tested. In particular,
54.84 GiB device occupancy does not imply a 54.84 GiB minimum VRAM requirement,
and 2.74 GiB tensor allocation alone does not prove that a 4 GiB GPU is sufficient.
