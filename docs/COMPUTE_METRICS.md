# Compute Metrics for Main Runs

This adds measurement only. It does not change PPO, observations, masks, reward,
episode grouping, seeds, or the qualified CPU-only parallel rollout protocol.
Historical runs are not modified. Missing historical peaks cannot be recovered
from model checkpoints or estimated from the current process.

## Recorded Metrics

| Metric | Definition |
| --- | --- |
| training_seconds | Sum of measured training update regions; excludes validation, checkpoint I/O, and update logging |
| rollout/wall_seconds | Collection, environment resets, policy inference, GAE, worker communication and batch concatenation |
| ppo_update/wall_seconds | PPO optimization; not pure GPU kernel time |
| simulator_ticks_per_second | Actual simulator ticks / measured seconds, not Blue decisions |
| cpu_seconds_sampled | Sum of sampled user+system CPU deltas across this process and descendants |
| cpu_mean_percent | 100 x sampled CPU seconds / wall seconds; 100% is one core, values above 100 are valid |
| cpu_peak_percent_sampled | Maximum sampled interval CPU usage, not an instantaneous peak |
| peak_process_tree_rss_bytes_sampled | Maximum sampled sum of resident RAM across parent and descendants |
| cuda_peak_allocated_bytes | PyTorch CUDA allocator peak allocated memory on the policy device, in the current process |
| cuda_peak_reserved_bytes | PyTorch CUDA allocator peak reserved memory on that device, in the current process |
| resource_read_errors | Failed resource queries; nonzero means resource coverage may be incomplete |

Training stores cumulative costs and maxima, both overall and separately for
rollout and PPO update. Evaluation stores a separate compute_metrics object per
mode/q/call, plus seconds_per_episode. Resets are inside the measured evaluation
region; environment construction, checkpoint loading and JSON/TensorBoard output
are outside that region.

Sampling interval is 0.2 seconds, with additional boundary samples. Short-lived
processes and short peaks can be missed. Summed RSS may double-count shared pages;
it is not unique physical RAM or system-wide memory usage. Idle workers still
contribute resident memory during periodic evaluation. Final evaluation occurs
after training workers have been closed. Keep these contexts separate.

CUDA measurements synchronize at region boundaries and reset allocator peak
counters at region entry. They include resident tensors present at entry and
cached allocator reservations. No empty_cache call is made. These values are not
total device VRAM, CUDA context memory, GPU utilization, energy, or GPU-hours.
CPU/MPS runs report CUDA metrics as null, not zero. CPU parallel training remains
CPU-only; adding these measurements does not qualify a CUDA training pipeline.

## Files and TensorBoard

- config.json: compute_metadata (platform, Python/Torch/psutil, CPU counts,
  system RAM, device/GPU name when CUDA, sampling protocol).
- training_history.jsonl, training_summary.json and checkpoints: compute metrics.
- evaluation JSON, including retention curves: compute_metrics and compute_metadata.
- compute_summary.json: written when the entire training/evaluation command
  completes. Includes training_with_validation_seconds, final_evaluation_wall_seconds
  and run_wall_seconds. These include I/O/setup in their respective command regions.
  run_wall_seconds starts after run-directory creation and ends before this file
  is written; it excludes interpreter startup and final logger cleanup.
- TensorBoard: training/compute/*, evaluation namespaces /compute/*, and
  compute_run/*. Null values are not plotted as zero.

The broader command timings overlap with the detailed timings: do not add them
all together. Evaluation throughput is simulator throughput, not inference-only
latency. Logging/sampling overhead is included where performed inside a region.

## Fair Comparisons

Compare encoder timing and memory on the same hardware/runtime, worker count,
Torch/BLAS thread limits, update grouping, evaluation protocol and background load.
Do not pool Mac, Kaggle and Colab timing measurements into a single encoder mean.
Report hardware groups separately and compute sample standard deviation across
training seeds within each group. GPU names, utilization percentages and peak
memory do not translate directly into monetary cost; record rental duration and
the actual bill separately if reporting cost.

Use the instrumented pipeline for new main runs only. Old 300k return/mission
results remain valid, but their unrecorded CPU/RAM/GPU metrics remain missing.
No temperature, power, GPU utilization or inference-only latency is collected.

## Validation

Tests cover stage accumulation versus peak maxima, worker CPU/RAM accounting,
missing metrics, thread cleanup, RNG and parameter invariance, JSON/TensorBoard
integration, and the device-specific CUDA API calls using mocks. Actual CUDA
measurement must be smoke-tested on an NVIDIA runtime before GPU cost reporting.
