# Team 14

Seven people built this robot for the SNU AI ROBOT CHALLENGE 2026. Every name links
to that person's GitHub account.

| Member | GitHub | Role |
|---|---|---|
| Yeonwoo Cho (조연우) | [@YenCho](https://github.com/YenCho) | Project lead · navigation and driving algorithms · Isaac Sim simulation · perception model fine-tuning |
| Jaeyoung Kim (김재영) | [@jaeyoungi2006](https://github.com/jaeyoungi2006) | Synthetic data generation · perception model training |
| Minjun Jang (장민준) | [@laufemj](https://github.com/laufemj) | Navigation and driving algorithms |
| Junhwan Lim (임준환) | [@Junhwaannn](https://github.com/Junhwaannn) | Robot hardware design |
| Minseok Kim (김민석) | [@kqwertyms](https://github.com/kqwertyms) | Robot hardware design |
| Junseo Kim (김준서) | [@rlawnstj06-snu](https://github.com/rlawnstj06-snu) | Robot hardware design |
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
Yeonwoo Cho ([@YenCho](https://github.com/YenCho)) and Minjun Jang
([@laufemj](https://github.com/laufemj)).

**Simulation** — the Isaac Sim kinematic surrogate used to validate the mission
logic before each field session was built by Yeonwoo Cho
([@YenCho](https://github.com/YenCho)).

## Tooling disclosure

Parts of this codebase were written with AI pair-programming assistance
(Claude Code and Codex). All design decisions, hardware, field testing and
competition operation were the team's own.
