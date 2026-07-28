import math

from robot_hardware.mecanum_control import (  # noqa: E402
    MecanumKinematics,
    MecanumOdometry,
    clamp_twist_components,
    parse_mecanum_state_line,
    parse_move_done_line,
    scale_wheel_speeds,
)

RADIUS = 0.034
HALF_LENGTH = 0.150
HALF_WIDTH = 0.125
CPR = 2464


def make_kinematics():
    return MecanumKinematics(RADIUS, HALF_LENGTH, HALF_WIDTH)


def test_forward_drives_all_wheels_equally():
    wheels = make_kinematics().body_to_wheels(0.2, 0.0, 0.0)
    assert all(abs(value - wheels[0]) < 1e-9 for value in wheels)
    assert wheels[0] > 0.0


def test_left_strafe_wheel_signs():
    fl, fr, rl, rr = make_kinematics().body_to_wheels(0.0, 0.2, 0.0)
    assert fl < 0.0
    assert fr > 0.0
    assert rl > 0.0
    assert rr < 0.0
    assert abs(fl + fr) < 1e-9
    assert abs(rl + rr) < 1e-9


def test_ccw_rotation_wheel_signs():
    fl, fr, rl, rr = make_kinematics().body_to_wheels(0.0, 0.0, 1.0)
    assert fl < 0.0
    assert rl < 0.0
    assert fr > 0.0
    assert rr > 0.0


def test_inverse_forward_roundtrip():
    kinematics = make_kinematics()
    for vx, vy, wz in [
        (0.2, 0.0, 0.0),
        (0.0, 0.15, 0.0),
        (0.0, 0.0, 0.8),
        (0.12, -0.08, 0.4),
    ]:
        wheels = kinematics.body_to_wheels(vx, vy, wz)
        recovered = kinematics.wheels_to_body(*wheels)
        assert abs(recovered[0] - vx) < 1e-9
        assert abs(recovered[1] - vy) < 1e-9
        assert abs(recovered[2] - wz) < 1e-9


def test_clamp_body_twist_scales_uniformly():
    kinematics = make_kinematics()
    vx, vy, wz = kinematics.clamp_body_twist(2.0, 1.0, 3.0, 8.0)
    wheels = kinematics.body_to_wheels(vx, vy, wz)
    peak = max(abs(value) for value in wheels)
    assert peak <= 8.0 + 1e-6
    # Direction is preserved.
    assert vx > 0.0 and vy > 0.0 and wz > 0.0
    assert abs(vy / vx - 0.5) < 1e-6


def test_odometry_forward():
    odometry = MecanumOdometry(make_kinematics(), CPR)
    odometry.update([0, 0, 0, 0])
    # One full wheel revolution forward on every wheel.
    odometry.update([CPR, CPR, CPR, CPR])
    assert abs(odometry.x - 2.0 * math.pi * RADIUS) < 1e-6
    assert abs(odometry.y) < 1e-9
    assert abs(odometry.yaw) < 1e-9


def test_odometry_strafe_left():
    odometry = MecanumOdometry(make_kinematics(), CPR)
    odometry.update([0, 0, 0, 0])
    odometry.update([-CPR, CPR, CPR, -CPR])
    assert abs(odometry.x) < 1e-9
    assert abs(odometry.y - 2.0 * math.pi * RADIUS) < 1e-6
    assert abs(odometry.yaw) < 1e-9


def test_odometry_rotation_in_place():
    kinematics = make_kinematics()
    odometry = MecanumOdometry(kinematics, CPR)
    odometry.update([0, 0, 0, 0])
    # Wheel angle for a 90 degree CCW body rotation.
    wheel_rad = (math.pi / 2.0) * kinematics.rotation_factor / RADIUS
    ticks = int(round(wheel_rad * CPR / (2.0 * math.pi)))
    odometry.update([-ticks, ticks, -ticks, ticks])
    assert abs(odometry.x) < 1e-3
    assert abs(odometry.y) < 1e-3
    assert abs(odometry.yaw - math.pi / 2.0) < 1e-2


def test_parse_mecanum_state_line():
    sample = parse_mecanum_state_line(
        'STATE,12345,10,-11,12,-13,100,-101,102,-103,1'
    )
    assert sample.device_ms == 12345
    assert sample.ticks == (10, -11, 12, -13)
    assert sample.pwm == (100, -101, 102, -103)
    assert sample.mode == 1
    assert parse_mecanum_state_line('OK m') is None
    assert parse_mecanum_state_line('STATE,1,2,3') is None
    assert parse_mecanum_state_line('STATE,x,2,3,4,5,6,7,8,9,0') is None


def test_parse_move_done_line():
    errors = parse_move_done_line('DONE,move,0.01,-0.02,0.003,0.0')
    assert errors == (0.01, -0.02, 0.003, 0.0)
    assert parse_move_done_line('DONE,other,1,2,3,4') is None
    assert parse_move_done_line('OK move') is None


def test_scale_wheel_speeds():
    wheels = scale_wheel_speeds((10.0, -20.0, 5.0, 0.0), 10.0)
    assert max(abs(value) for value in wheels) <= 10.0 + 1e-9
    assert abs(wheels[0] / wheels[1] + 0.5) < 1e-9
    unchanged = scale_wheel_speeds((1.0, 2.0, 3.0, 4.0), 10.0)
    assert unchanged == (1.0, 2.0, 3.0, 4.0)


def test_clamp_twist_components():
    assert clamp_twist_components(1.0, -1.0, 2.0, 0.3, 0.2, 0.8) == (
        0.3,
        -0.2,
        0.8,
    )
