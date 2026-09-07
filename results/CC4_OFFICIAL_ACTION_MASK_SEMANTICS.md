# CAGE 4 Official Action-Mask Semantics

## Result

The official mask does not expose changing compromise or evidence state: it is constructed at reset and remains fixed within the episode. It does expose static random-host existence/session information through valid host-target action slots.

Temporal changes detected: `0`.
Mask checked while an action was in progress: `True`.

| Agent | Unique masks across episodes | Valid actions | Existing hosts | Correlation |
|---|---:|---:|---:|---:|
| blue_agent_0 | 25 | 34-82 | 16-16 | N/A (constant host count) |
| blue_agent_1 | 20 | 42-82 | 16-16 | N/A (constant host count) |
| blue_agent_2 | 22 | 34-82 | 16-16 | N/A (constant host count) |
| blue_agent_3 | 21 | 34-74 | 16-16 | N/A (constant host count) |
| blue_agent_4 | 30 | 130-202 | 48-48 | N/A (constant host count) |

## Consequence

A zero-observation policy that retains this mask is not information-free: it still receives static topology/existence information. Mask-only and no-mask gates must therefore be reported together.
