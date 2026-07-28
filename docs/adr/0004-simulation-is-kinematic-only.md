# ADR-0004 — The simulation is kinematic only

**Status:** Accepted (2026-07-07). Held for the rest of the project.

## Context

The Isaac Sim scene can run either way: `--drive-mode physics` articulates four mecanum
wheels with Kaya-style 45° passive sphere rollers and lets the holonomic behaviour emerge
from contact, or `--drive-mode kinematic` integrates the commanded body twist and moves the
root directly. Physics mode is the more impressive demo, and it is the one that tempts you
into believing the simulator can answer questions about the real robot.

It cannot, for a specific reason. At physics rates the real-time factor drops and CPU
contention rises, so the localiser — the same node the robot runs — executes under timing
conditions the robot never experiences. A pass or a fail in that regime is not evidence
about the robot; it is evidence about a different machine.

Meanwhile the questions the simulator *is* uniquely good at — does the mission state machine
handle a contradiction, does the router keep clearance, does the recognition chain snap to
the right grid cell — do not need contact physics at all.

## Decision

The simulator is a **kinematic surrogate for perception and mission logic**. Roller physics,
traction, grasp force and timing margins are validated on the real robot (MK4) and nowhere
else. Every published result from the simulator is a kinematic-mode result.

## Consequences

**Benefits.** Sessions run headless for hours at usable real-time factors. The full pipeline
rehearsal on 2026-07-07 collected 7 of 7 objects with 7 of 7 inside the storage rim and zero
non-target disturbances in 218–230 s, and reproduced on a fresh layout with a shuffled
target set using identical parameters. The mission code that ran there is byte-identical to
the code that ran on the robot.

**Costs, and they are not small.**

- **Grasping is faked.** On `CLOSE` the nearest object inside a hard-coded zone
  (0.095–0.245 m ahead of base centre, |lateral| ≤ 0.06 m) is attached to the robot. This
  validates *aiming* and never *holding*. Nothing about finger geometry, closing force or an
  icosahedron squirting out of the fingers is testable here.
- **Timing does not transfer.** Sim `move_relative` is a P-controller capped at 0.18 m/s
  against the robot's firmware trapezoid; measured moves ran a consistent **2.4–2.9×**
  slower. The surrogate cannot tell you whether a strategy fits in 180 s. Final 2 was lost
  to the 3-minute timeout, and no amount of simulation would have predicted it.
- **The failures that actually cost us points were outside its scope.** Final 1 was lost to
  the OpenRB going silent on the real robot - the bridge process stayed up. A kinematic surrogate with a cheat grip has
  no way to surface that class of fault.
- **Perception parity is conditional and had to be earned.** Getting the renderer honest
  took three separate fixes, one of which had been silently capping detection range at
  1.9 m. Face-identity accuracy in sim remains optimistic relative to printed paper under
  arena light. See [`simulation/docs/field-parity.md`](../../simulation/docs/field-parity.md).
- **The physics path is dead weight in the repository.** It is maintained enough to run and
  not enough to trust, which is the worst state for code to be in. It ships because deleting
  it would make re-deciding this ADR expensive; it should be read as unvalidated.
