# CybORG++ CAGE-2 Patch Provenance

The project keeps the current CybORG API and ports the six published fixes from
the CybORG++ `Debugged_CybORG` tree into the vendored simulator instead of
replacing the full runtime with the older fork.

## Upstream

- Repository: https://github.com/alan-turing-institute/CybORG_plus_plus
- Retrieved commit: `c343476cb4490ba3e850561e0d11e7036ce50822`
- Retrieved date: `2026-08-09`
- Source notes: `Debugged_CybORG/README.md`

## Ported Fixes

1. Return a failed `Remove` result when no suspicious process is stopped, and
   propagate the `StopProcess` result when one is attempted.
2. Use port 21 for the Vsftpd decoy instead of port 80.
3. Mark the SMB process in `Internal_image.yaml` with `Process Type: smb`.
4. Use port 3390 for MySQL in `linux_user_host_image1.yaml`.
5. Count all published compromise usernames in `PwnRewardCalculator`.
6. Classify an exploit marker on remote port 4444 before applying scan-shape
   rules in `BlueTableWrapper`.

## Upstream SHA-256

```text
457f696e5faf17c6166752e6684d95d6ba904b96f55cc3b2817a4c144830eede  Debugged_CybORG/README.md
565040df2bd0e33ceac2f3c9c22adc6763c7ee5a609396093c71031dad48dfb2  Debugged_CybORG/CybORG/CybORG/Shared/Actions/AbstractActions/Remove.py
918778d2aa40cf567fd14ab349d1ebdbc346c73d121f3356595e70b1afc047ac  Debugged_CybORG/CybORG/CybORG/Shared/Actions/ConcreteActions/DecoyVsftpd.py
87ad65c96ed44a2338f5df3ed00595156195b059f55df1b6dc9ec6269b3b303f  Debugged_CybORG/CybORG/CybORG/Shared/Scenarios/images/Internal_image.yaml
d4b14899e88015c3d16de616b2a68037a3ed09d52c31192fe33cb071154bd6e3  Debugged_CybORG/CybORG/CybORG/Shared/Scenarios/images/linux_user_host_image1.yaml
0a9228f9898eab892c923da5aafa20740349853a85d2a01267c73221933c1de6  Debugged_CybORG/CybORG/CybORG/Shared/RedRewardCalculator.py
1c7540f77ad4ff259150298642f406a8153796b468f9ed33a79c9f6f12c6ffc8  Debugged_CybORG/CybORG/CybORG/Agents/Wrappers/BlueTableWrapper.py
```

## Validation

Run the focused regression suite with:

```bash
uv run --python 3.10 pytest tests/test_cyborg_plus_plus_patches.py -q
```

## Local Restore Service Invariant

The current runtime stores each service's process identifier under `process`,
but `Host.create_backup()` and `Host.restore()` previously read the obsolete
`PID` key. A Restore therefore reconstructed `OTService` with
`process: null`, even though the process itself still existed. This made the
operational availability metric disagree with the simulator's native reward.

The local fix preserves the `process` mapping across backup and Restore. It
does not change Restore duration, cost, session removal, or reward values. This
fix is project-local and is not represented as one of the six published
CybORG++ changes.

Together these changes define
`cyborg_plus_plus_bugfixes_v1_restore_service_invariant_v1`. They do not import
MiniCAGE, alter the benchmark reward, or change the topology.
