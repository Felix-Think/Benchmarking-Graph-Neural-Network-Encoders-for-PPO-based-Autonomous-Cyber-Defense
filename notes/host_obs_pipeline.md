# Host Observation Pipeline

Trang thai: **active cho Pilot**.

## Data flow

```text
raw native Blue observation
  -> RawBlueObservationAdapter
  -> 16 x 26 shared node-feature matrix
  -> exact fixed-budget retention mask
  -> optional evaluator-only observation intervention
  -> Flat reshape OR GraphSAGE graph input
```

`RawBlueObservationAdapter` nam trong
`src/acd_benchmark/wrappers/shared/observation.py`. Adapter chi dung:

- static inventory ma Blue nhan tai reset;
- static topology prior da audit;
- raw observation cua timestep hien tai;
- reset baseline de phan biet process/file/session anomaly.

Adapter khong merge sparse observation qua thoi gian, khong doc `success`,
khong nhan selected action va khong doc true simulator state.

## Retention contract

Mau so la 13 actionable hosts, khong phai 16 entities. Ba router chi co static
features. Fixed budgets tai q=1/.75/.5/.25 lan luot la 13/10/7/3, stratified
theo User/Enterprise/Operational.

Khi mot host bi an, tat ca raw telemetry cua host, ke ca
`telemetry.native_evidence`, bi dua ve 0. Chi
`missingness.wrapper_retained=0` cho policy biet telemetry khong co san. Dieu
nay tranh lam lo vi tri event chi qua viec host co xuat hien trong raw native
observation.

## Representation parity

Flat va GraphSAGE nhan dung cung mot realization cua node-feature matrix. Flat
reshape theo canonical node order va khong nhan adjacency. GraphSAGE nhan them
`edge_index` tu static topology; day la treatment can duoc ghi ro, khong phai
information parity tuyet doi.
