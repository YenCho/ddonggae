# Team 14

Seven people built this robot for the SNU AI ROBOT CHALLENGE 2026. Every name links
to that person's GitHub account.

| Member | GitHub | Primary Contributions |
|---|---|---|
| Yeonwoo Cho (조연우) | [@YenCho](https://github.com/YenCho) | Project lead · navigation and driving algorithms · Isaac Sim simulation · perception model fine-tuning |
| Jaeyoung Kim (김재영) | [@jaeyoungi2006](https://github.com/jaeyoungi2006) | Synthetic data generation · perception model training |
| Minjoon Jang (장민준) | [@laufemj](https://github.com/laufemj) | Navigation and driving algorithms · robustness and reliability improvement |
| Junhwan Lim (임준환) | [@Junhwaannn](https://github.com/Junhwaannn) | Robot hardware design |
| Minseok Kim (김민석) | [@kqwertyms](https://github.com/kqwertyms) | Robot hardware design |
| Junseo Kim (김준서) | [@rlawnstj06-snu](https://github.com/rlawnstj06-snu) | Robot hardware design · robustness and reliability improvement |
| Seojun Han (한서준) | [@HSJ1127](https://github.com/HSJ1127) | — |

## How the work split

**Hardware** — the mecanum chassis, the camera mast, the parallel gripper and the
wiring were designed and built by Junhwan Lim
([@Junhwaannn](https://github.com/Junhwaannn)), Minseok Kim
([@kqwertyms](https://github.com/kqwertyms)) and Junseo Kim
([@rlawnstj06-snu](https://github.com/rlawnstj06-snu)).

**Perception** — the entire training set is synthetic. The Blender
data-generation pipeline and the model training were done by Jaeyoung Kim
([@jaeyoungi2006](https://github.com/jaeyoungi2006)) in a separate repository;
Yeonwoo Cho ([@YenCho](https://github.com/YenCho)) handled fine-tuning against
real arena captures and on-robot deployment.

**Navigation** — the Nav2-free localisation and control stack was designed by
Yeonwoo Cho ([@YenCho](https://github.com/YenCho)) and Minjoon Jang
([@laufemj](https://github.com/laufemj)).

**Simulation** — the Isaac Sim kinematic surrogate used to validate the mission
logic before each field session was built by Yeonwoo Cho
([@YenCho](https://github.com/YenCho)).

**Robustness and reliability** — making the robot survive its own bad days, worked
on by Minjoon Jang ([@laufemj](https://github.com/laufemj)) and Junseo Kim
([@rlawnstj06-snu](https://github.com/rlawnstj06-snu)). Two halves to it. The
hardware half: serial drop-outs, respawn and self-heal on the two microcontroller
bridges, the USB wiring that stopped them dropping in the first place, and the
bring-up health checks that refuse to declare a half-alive stack ready. The
decision half: what the robot does when the centre scan comes back thin or
self-contradictory — checking the grid map for consistency, deciding which cells
are trustworthy enough to commit a 25-second round trip to, and ranking the pickup
order by expected value (20-point fruit against 10-point shapes, against the 2×
penalty for a mispick) rather than by distance. That ranking is what both finals
ran on.

## Tooling disclosure

Parts of this codebase were written with AI pair-programming assistance
(Claude Code and Codex). All design decisions, hardware, field testing and
competition operation were the team's own.
