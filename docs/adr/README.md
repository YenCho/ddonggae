# Architecture Decision Records

Five decisions shaped this robot more than any other. Each record states the situation that
forced the choice, what was chosen, and what it cost — the costs are the part worth reading.
All of them are grounded in measurements or failures from this project, not in general
principle.

| # | Decision | Status | One-line summary |
|---|---|---|---|
| [0001](0001-drop-nav2.md) | Drop Nav2 | Accepted 2026-07-04 | Full scan matching cost 345–449 ms in an empty known square; a hand-written wall-range matcher does it in 8–10 ms offline / ~41 ms on the Jetson — at the price of no planner, no obstacle avoidance and no reusable stack. |
| [0002](0002-json-over-std-msgs-instead-of-custom-messages.md) | JSON over `std_msgs/String` instead of custom messages | Accepted, with a known failure | Every payload is echo-able and pub-able from a field terminal; the interface package stayed empty. A misread key (`grasp_state` vs `state`) then made every successful grasp score as a failure. |
| [0003](0003-firmware-owns-motion-primitives.md) | Firmware owns the motion primitives | Accepted 2026-07-14 | The Arduino runs the velocity PID and a synchronised 4-wheel trapezoidal position profile, so moves are immune to host load — but deadlines, state and buffer limits now live across a serial link. |
| [0004](0004-simulation-is-kinematic-only.md) | The simulation is kinematic only | Accepted 2026-07-07 | Isaac Sim is a surrogate for perception and mission logic; physics was validated on the real robot. Grasping in sim is faked and sim timing runs 2.4–2.9× slow. |
| [0005](0005-synthetic-only-training-data.md) | Synthetic-only training data | Accepted, later amended | Zero hand-labelled images; everything rendered in Blender. It survived a rule change four days before the competition, and it lost the qualifiers to an under-ripe apple. |

## Format

Each record is Context / Decision / Consequences / Status. Consequences always list costs,
including the ones we paid publicly. If a decision was later amended, the amendment is in
the Status line rather than in a new record — this is a frozen project, not a living
codebase.

## Related reading

- [`docs/02-architecture.md`](../02-architecture.md) — how the three parts compose.
- [`docs/07-results-and-lessons.md`](../07-results-and-lessons.md) — measured outcomes and
  the three competition failures.
- [`simulation/docs/field-parity.md`](../../simulation/docs/field-parity.md) — the evidence
  behind ADR-0004.
