# ADR-0002 — JSON over `std_msgs/String` instead of custom messages

**Status:** Accepted. Shipped in the competition build, with the failure in §Consequences
acknowledged.

## Context

The system needs structured payloads on about a dozen topics: goals and pose seeds, motor
telemetry, gripper and mast commands and state, relative-move requests and their results.
The ROS 2 idiom is a `.msg` definition in an interface package. We created that package
(`robot_interfaces`) and it stayed empty for the life of the project.

The constraints that pushed against it: the robot is field-debugged from a laptop terminal,
often with a 3-minute window between rounds; payload shapes changed almost daily as field
observations added fields (`max_v` on a move, a settling metric on a grasp); and a
significant part of the graph is not ROS packages at all — the match runner is a plain
rclpy process, and the Isaac bridge lives inside Isaac's own Python.

## Decision

Every structured payload is a `std_msgs/String` carrying JSON. This covers
`/arena_lightweight/{goal,pose,control,status}`, `/motor/state`,
`/gripper/{command,state,grasp}`, `/lift/{command,state}` and
`/base/{move_relative,move_result}`.

## Consequences

**Benefits.**

- No interface package to build, and no rebuild-and-resource cycle when a payload gains a
  field. Adding `max_v` to a relative move
  (`hardware/ros2/robot_hardware/robot_hardware/mecanum_bridge_node.py:398`) was a one-sided
  change: old senders keep working.
- Every command in the system can be issued and inspected from a shell with
  `ros2 topic pub` / `ros2 topic echo`, with no custom message on the path. During a field
  round this is the difference between diagnosing something and not.
- Bags replay on any machine without the interface package installed.
- The simulator can mirror the robot's contract exactly by emitting the same strings —
  which is what lets `mission/match_runner.py` drive sim and hardware unmodified
  ([ADR-0004](0004-simulation-is-kinematic-only.md)).

**Costs.**

- **There is no schema, so there is no schema check.** Every consumer hand-rolls
  `json.loads` inside a `try/except` (`arena_control_node.py:636`, `:1055`, `:1104`;
  `gripper_bridge_node.py:716`). A typo in a key is not an error; it is a default value.
- **This cost us a real failure.** Until the early hours of 2026-07-20 the match runner read
  `grasp_state` from `/gripper/grasp` while the bridge published `state`. The read always
  returned `""`, so **every successful grasp was scored as a failure** and the robot kept
  re-grasping objects it was already holding. The fix is one line and a fallback
  (`perception/fieldlib.py:571-578`). A typed message would have made this impossible to
  express.
- No `rqt_plot`, PlotJuggler or type introspection on any payload — these topics are opaque
  strings to every standard ROS tool.
- Strings are larger on the wire and re-parsed independently by every subscriber.

If this project ran a second season, the honest answer is: keep JSON for the operator-facing
and cross-runtime topics, and put the grasp/move result contract — the ones a silent misread
can lose a match on — into typed messages.
