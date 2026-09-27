# CPU Rollout + CUDA Learner

## Execution Contract

- Two spawned CPU workers collect four whole episodes per PPO update.
- Workers use an immutable CPU snapshot of the learner. No asynchronous/stale-policy training.
- CybORG, canonical observations, masks, action sampling and SMDP GAE run on CPU.
- PPO transfers the required batch tensors to CUDA once per update, then runs forward/backward and Adam on CUDA.
- After the update, the next collection receives the new weights.
- Evaluation uses a synchronized CPU copy, on the same host as the learner.
- Precision is FP32; TF32 is disabled. No AMP, reward, model, parameter budget, batch-size or masking changes.
- Action RNG remains per_episode_v1 on CPU. CUDA optimization need not be bitwise identical to CPU optimization.
- CUDA is mandatory for the new configuration: unavailable CUDA fails before creating a run.

The CPU path remains for local tests and reading old results, not as an extra benchmark condition.
The existing 300k CPU runs are not modified or appended to.

## Open The Notebook

Open notebooks/colab_cuda.ipynb on Colab. It clones GitHub at a pinned source commit;
no source upload is needed. For a private repository, put a read-only fine-grained
token in Colab Secrets as GITHUB_TOKEN and enable notebook access. Never paste it
in a code cell, URL or output. The token is passed only to the clone subprocess.

## Install And Qualify

Choose an NVIDIA GPU runtime first. Use the locked Python 3.10 environment, not
Colab's preinstalled Torch/Python. The existing lock includes Linux CUDA 12.1
dependencies for Torch 2.2.0. Do not silently upgrade Torch or PyG for one encoder.

```bash
python -m pip install uv
uv sync --locked --python 3.10 --group dev
uv run --no-sync python -m scripts.experiments.smoke_cuda
```

The smoke test must pass on the actual GPU. It checks all three encoders,
CPU-worker log probabilities versus CUDA learner log probabilities before updates,
finite gradients/metrics, changed learner weights, optimizer device, checkpoints,
CPU evaluation and GPU memory reporting. Without CUDA it fails, not skips.

This is a correctness gate, not evidence of learning or speedup. CPU rollout can
still dominate elapsed time; GPU acceleration affects the learner only.

## Run The Queue

```bash
uv run --no-sync python -u -m scripts.experiments.run_encoder_matrix \
  --config configs/experiments/encoder_retention_500k_cuda.json \
  --run-root /content/drive/MyDrive/acd_runs/encoder_retention_500k_cuda_seed42 \
  --seeds 42 \
  --foreground
```

Repeat with seeds 43, 44, 45 and 46 using different run roots. Each seed runs six
configurations sequentially: NodeMLP/GCN/GraphSAGE x Full/mixed-q.
Omit --seeds to run all 30 sequentially if the session can accommodate it.
Without --foreground or --launch, the command only prints the plan.

Budget is 500k simulator ticks per policy, with validation at 100k/200k/300k/400k/500k.
There are 30 runs (15M nominal training ticks); evaluation is additional compute.
Evaluation seeds, q values, persistent masking, four episodes/update, two CPU
workers and PPO hyperparameters remain unchanged from the prior experiment.
The old 300k configurations and launchers have been retired; historical run
manifests and source snapshots remain with their original results.

Outputs are written directly to the chosen run root, including console.log per
cell, status.json, source snapshot, checkpoints, JSON and TensorBoard events.
Foreground mode prints cell starts; inspect the current cell's console.log for updates.
Google Drive I/O contributes to end-to-end wall time; use the same storage setup
for all cells. Rollout/PPO stage timing is also reported separately.

## Interruption And Hardware

Use RESUME=True with the same SEED, RUN_TAG, source commit and output root.
The queue skips finished cells and restores latest_checkpoint.pt for an interrupted
cell. It refuses changed source/config instead of silently restarting. See RESUME.md.
Do not run two notebooks against the same root. Legacy runs without a committed
latest checkpoint are not supported for resume.

For time/resource comparisons use the same GPU model, CPU allocation, worker
count, precision, dependencies and storage policy. Compare the runtime metadata
between sessions before pooling timing. Colab does not guarantee a particular
GPU or unlimited runtime, even for paid plans.

Each config.json records GPU model/VRAM/compute capability, CUDA and Torch,
CPU counts, RAM and thread count. compute_summary.json separates rollout,
PPO and evaluation costs; peak CUDA allocated/reserved bytes are PyTorch allocator
peaks, not total GPU board memory or GPU utilization. CPU/RAM metrics sample the
process tree and may miss short peaks. See COMPUTE_METRICS.md.

CPU evaluation reports null CUDA allocator metrics. The learner still resides
on GPU during validation, so this is not a claim of zero GPU allocation for the
whole process. Training allocator peaks include the resident learner.

CUDA scatter operations in GNNs can be nondeterministic. Seeds are recorded;
bitwise reproducibility across hardware is not promised. Do not combine old M4
timing and new CUDA timing in one hardware-normalized summary.

## Sources

- [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyTorch previous-version installation](https://pytorch.org/get-started/previous-versions/)
- [Colab resource limits and GPU availability](https://research.google.com/colaboratory/faq.html)

## Resume Và Log Tiến Độ

Xem [RESUME.md](docs/RESUME.md). Notebook dùng RESUME=True để tiếp tục cùng run.
latest_checkpoint.pt lưu sau mỗi PPO update và validation hoàn tất; không tiếp tục
giữa một episode. Log mỗi50 episode ticks; PPO loss/KL in sau mỗi update.
Run cũ không có checkpoint phục hồi mới phải giữ riêng, không tự chạy lại từ đầu.
