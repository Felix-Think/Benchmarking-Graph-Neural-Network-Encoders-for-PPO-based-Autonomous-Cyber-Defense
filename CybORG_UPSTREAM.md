# Local CybORG patches

Thu muc `CybORG/` la vendored simulator va co cac ban va cuc bo da duoc kiem
thu. Khong thay CybORG bang mot checkout upstream moi ma khong port va chay lai
cac regression test duoi day.

## SSH exploit-decoy event

File:
`CybORG/CybORG/Simulator/Actions/ConcreteActions/ExploitActions/SSHBruteForce.py`

`SSHBruteForce` truyen `target_host` vao helper va ghi network-connection event
vao host dung khi cham exploit decoy. Regression test:
`tests/test_cyborg_ssh_decoy_fix.py`.

## StopProcess service-session reload

File: `CybORG/CybORG/Simulator/Actions/ConcreteActions/StopProcess.py`

`StopProcess` resolve session ID thanh session object truoc khi dung username,
session type, agent, parent va timeout de reload service session. Regression
test: `tests/test_cyborg_stop_process_fix.py`.

## Validation

```bash
uv run --python 3.10 pytest tests/test_cyborg_ssh_decoy_fix.py \
  tests/test_cyborg_stop_process_fix.py -q
```

## CybORG++ CAGE-2 bug fixes

Sau khi doi chieu voi `alan-turing-institute/CybORG_plus_plus` commit
`c343476cb4490ba3e850561e0d11e7036ce50822`, sau ban va cong bo trong
`Debugged_CybORG` da duoc port vao API CybORG hien tai. Khong thay nguyen runtime
bang fork cu vi benchmark dang dung scenario generator va wrapper cua API moi.

Nguon, checksum va mapping chi tiet nam tai:
`patches/cyborg_plus_plus/README.md`.

Regression coverage:

```bash
uv run --python 3.10 pytest tests/test_cyborg_plus_plus_patches.py -q
```

## Restore service-process invariant

File: `CybORG/CybORG/Simulator/Host.py`

Runtime services use the key `process`, while the backup/restore path read the
obsolete key `PID`. The local patch preserves the service-to-process mapping,
so `Restore(Op_Server0)` no longer leaves `OTService` pointing to `None`.
This is a local simulator invariant fix, not a claimed CybORG++ upstream patch.
It does not change native reward or Restore cost. Regression coverage is part
of `tests/test_cyborg_plus_plus_patches.py`.
