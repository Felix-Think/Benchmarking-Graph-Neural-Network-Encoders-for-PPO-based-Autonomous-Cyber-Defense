# Flat Native CAGE-2 Learning Qualification

Protocol authority: `configs/pilot.yaml`, identity
`pilot_cc2pp_flat_native_learning_qualification_v2`.

## Decision Boundary

Day la learning qualification cho Flat, khong phai encoder benchmark. GraphSAGE,
mixed-q, partial-observation curve va topology generalisation bi khoa cho den
khi Flat hoc duoc policy quan sat phu thuoc tren native CAGE-2 q=1.

Protocol chi thay doi hai module da duoc cong khai trong nghien cuu truoc:

1. Cardiff-style action abstraction;
2. observable temporal state estimation lay cam hung tu POMDP/BF-PPO.

Native `HybridAvailabilityConfidentiality` reward duoc giu nguyen. Khong reward
shaping, semantic action mask, learned normalizer, forced opening actions hoac
Red-agent fingerprinting.

## Action Contract

Cardiff cong bo final PPO action surface gom 27 action
`Analyse/Remove/Restore` tren chin host va chin greedy-decoy macros. Pilot dung
dung policy surface 36 action nay. Moi `DeployDecoy(host)` duoc resolve theo
greedy order cong bo cua Cardiff; Restore xoa predicted decoy history tren host;
khi danh sach day, macro fallback sang `Remove(host)`.

Resolver chi doc action policy da chon trong episode. No khong doc simulator
true state, action success hoac evaluator metrics. Manifest ghi ca policy action
va concrete native CybORG action de co the audit performance confound.

Source: <https://github.com/john-cardiff/-cyborg-cage-2>

## State-Estimation Contract

Policy input giu raw current-step Blue telemetry va them mot deterministic,
observable filter:

- five-state host security belief: unseen, clean, uncertain, user, privileged;
- scan-seen memory;
- latest-scan marker;
- predicted decoy-fill fraction.

Update chi dung exact Blue observation ma policy nhan va selected action history.
Khong co true state, top-level success result hoac pre-mask hidden telemetry.

Day khong phai reproduction cua BF-PPO. BF-PPO dung particle filter, simulator
transition model va representative sampled hidden state. Pilot chi ke thua y
tuong recursive action-observation belief de co mot proxy nho, minh bach va co
the dung chung sau nay. Neu can so sanh BF-PPO that, no phai la baseline rieng,
khong duoc gan nhan cho wrapper nay.

Source: <https://arxiv.org/abs/2509.06539>

## Native-q1 Contract

- official Scenario2, RedMeanderAgent, horizon 100;
- 13 actionable hosts + 3 routers;
- q chi co `1.0`; all 13 host telemetry entries duoc retained;
- khong train hay evaluate q=.75/.50/.25 trong debug stage;
- static graph van duoc build de audit contract, nhung GraphSAGE khong train;
- native reward, CybORG++ bug-fix patchset, khong reward shaping.

q=1 van la native partial Blue view cua CAGE-2, khong phai simulator true state.

## PPO Contract

PPO dung MLP 2 x 64. Rollout moi update la 10,000 environment steps, 10 epochs,
learning rate 5e-4, Adam beta2=.99, gamma=.99 va clip=.2. Cac gia tri chinh
theo BF-PPO Appendix; GAE, entropy coefficient, value coefficient, gradient
clip va target-KL la implementation controls cua pipeline nay va duoc manifest
day du.

## Staged Gates

- 100k: smoke-learning checkpoint; kiem tra finite loss, KL va action collapse.
- 300k: chi tiep tuc neu return dang cai thien, policy khong trung open-loop
  Restore/best-fixed control va logits phan ung voi observation.
- 1M: final Flat qualification tren 100 held-out episodes.

Flat PASS khi:

- tot hon random;
- khong bi best fixed action hoac Restore round-robin Pareto-dominate tren
  native return, mission availability va privileged-compromise burden;
- khong collapse thanh mot action-target open-loop;
- pass q=1 observation-use gate;
- PPO diagnostics finite va target-KL early-stop fraction <=20%;
- policy/native action resolution, reward components va artifacts day du.

Neu Flat FAIL, dung lai va chan GraphSAGE. Khong sua reward de ep PASS. Neu Flat
PASS, freeze action/state/PPO contracts roi moi them GraphSAGE va mixed-q curve
tro lai bang protocol moi.

## Commands

```bash
uv run --python 3.10 python -m scripts.pilot.audit_protocol \
  --config configs/pilot.yaml --no-write

uv run --python 3.10 python -m scripts.pilot.run_matrix \
  --config configs/pilot.yaml --smoke --models flat --seeds 0 \
  --skip-baselines --skip-observation-gates --overwrite
```

Khong chay 1M truoc khi smoke, baseline calibration va 100k/300k review dat.
