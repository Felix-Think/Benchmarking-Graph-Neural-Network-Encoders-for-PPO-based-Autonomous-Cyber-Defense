# Resume At Committed PPO Updates

Use the same source commit, locked environment, seed, regime, hyperparameters,
episode/update grouping and nominal budget. Do not extend a finished budget by
calling this resume mode: the earlier partial final update changes grouping.

## Colab

1. Keep SEED and RUN_TAG unchanged.
2. Set RESUME=True, reconnect GPU, clone the pinned source, install, run CUDA tests,
   and mount the same Drive.
3. Launch the queue. Finished cells are skipped; an interrupted cell restores
   latest_checkpoint.pt. A not-yet-started cell starts normally.
4. Do not run another notebook with the same output root.

CLI: append --resume --foreground to scripts.experiments.run_encoder_matrix
with the original --run-root. For a standalone cell use scripts.training.train
with all original arguments plus --resume.

## Recovery Contract

- latest_checkpoint.pt exists before the first rollout, then is atomically replaced
  after each completed PPO update AND its periodic/milestone validation callbacks.
- Policy, optimizer, Python/NumPy/Torch/CUDA RNG, episode index, counters,
  training history, cyber metrics and validation/best selection are restored.
- Workers restart at episode boundaries using per_episode_v1 seeds.
- A crash during rollout, optimizer update or validation discards that uncommitted
  work. It is replayed from the previous committed update. Never save partially
  updated model weights as a completed update.
- Speculative JSON/best-checkpoint/milestone outputs are moved into discarded_attempts
  before restoring committed history. TensorBoard purges steps after the boundary.
- A crash during final evaluation resumes the completed training checkpoint and
  reruns final evaluation without more PPO updates.
- console.log appends across attempts; repeated rollout lines are expected.
  They are not additional committed training samples.
- Same-root process locking prevents concurrent queues on the same host.
  Do not rely on it as distributed coordination across separate Colab VMs.

## Legacy And Hardware Limits

Old e2a55dc runs lack the new committed latest checkpoint. They cannot be silently
converted into resumable runs. Preserve their outputs and explicitly start a new
run root if needed. A milestone/best checkpoint is not interchangeable with latest.

CPU tests compare uninterrupted versus resumed parameters, optimizer, RNG, metrics
and mixed-q episodes for all three encoders. CUDA resume checks are in smoke_cuda
and must pass on actual hardware. CUDA/GNN nondeterminism and hardware changes
mean cross-device bitwise equivalence is not promised.

Compute counters restored from the checkpoint cover committed work only, not
replayed work or downtime. compute_summary wall times cover the current session.
resume_sessions.jsonl records hardware per resume. Do not report these times as
the total cost of an interrupted experiment; label interrupted timing separately.

## Progress

progress_every_ticks=50 prints at episode ticks50/100/... and episode end on each
worker: episode seed/tick, committed global ticks, current raw reward, cumulative
episode return, OT availability, compromise burden, compromised hosts, successful
Impacts, completed TTR and unrecovered count. NA means undefined, not zero.
These are partial-episode metrics, not evaluation scores.

Two workers can print in interleaved order. committed_ticks advances only after
PPO updates, not for each worker tick. Loss, entropy, KL and update-level metrics
print after every update; they cannot be recomputed meaningfully every50 simulator
ticks without changing PPO. Plain flushed lines keep Colab/console logs readable;
no tqdm redraw or progress-monitor loop modifies environment/policy RNG.
