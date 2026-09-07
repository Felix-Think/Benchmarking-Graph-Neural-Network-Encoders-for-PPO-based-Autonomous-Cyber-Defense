# GCN versus Flat Feature-Gate Diagnostic

## Protocol warning

This is not a fair final encoder benchmark. GCN used
100000 requested simulator steps,
while Flat used 30000. The
parameter counts are approximately matched (291735
versus 292339). All comparisons use the
same 30 held-out environment seeds.

| Comparison | Mean paired delta | Bootstrap 95% CI |
|---|---:|---:|
| `gcn_minus_flat_full_return` | -33.07 | [-191.70, 121.23] |
| `gcn_minus_flat_static_return` | -10.90 | [-164.10, 139.60] |
| `dynamic_use_interaction_gcn_minus_flat` | -22.17 | [-193.70, 166.60] |
| `adjacency_use_interaction_gcn_minus_flat` | 91.30 | [-156.90, 343.70] |

The dynamic interaction is
`(GCN full - GCN static) - (Flat full - Flat static)`. The adjacency interaction
uses the analogous topology versus topology-without-adjacency difference.
