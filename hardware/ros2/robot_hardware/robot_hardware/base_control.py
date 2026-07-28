import csv
import math
from dataclasses import dataclass
from pathlib import Path

PWM_CALIBRATION_LABELS = (
    'left_forward',
    'left_reverse',
    'right_forward',
    'right_reverse',
)


@dataclass(frozen=True)
class EncoderSample:
    device_ms: int
    left_pwm: int
    right_pwm: int
    left_ticks: int
    right_ticks: int


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_encoder_line(line):
    fields = line.strip().split(',')
    if len(fields) < 7 or fields[0] != 'DATA':
        return None

    try:
        return EncoderSample(
            device_ms=int(fields[1]),
            left_pwm=int(fields[2]),
            right_pwm=int(fields[3]),
            left_ticks=int(fields[4]),
            right_ticks=int(fields[5]),
        )
    except ValueError:
        return None


def twist_to_wheel_speeds(
    linear_mps,
    angular_rps,
    wheel_radius_m,
    wheel_separation_m,
):
    half_track = wheel_separation_m * 0.5
    left = (linear_mps - angular_rps * half_track) / wheel_radius_m
    right = (linear_mps + angular_rps * half_track) / wheel_radius_m
    return left, right


class WheelPiController:
    def __init__(
        self,
        min_pwm,
        max_pwm,
        feedforward_slope,
        kp,
        ki,
        integral_limit,
        scale=1.0,
        positive_curve=None,
        negative_curve=None,
    ):
        self.min_pwm = float(min_pwm)
        self.max_pwm = float(max_pwm)
        self.feedforward_slope = float(feedforward_slope)
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral_limit = float(integral_limit)
        self.scale = float(scale)
        self.positive_curve = positive_curve
        self.negative_curve = negative_curve
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def feedforward(self, target_rad_s):
        if abs(target_rad_s) < 1e-6:
            return 0.0
        curve = (
            self.positive_curve
            if target_rad_s > 0.0
            else self.negative_curve
        )
        if curve is not None:
            pwm = curve.pwm_for_speed(abs(target_rad_s))
            return math.copysign(pwm * self.scale, target_rad_s)
        magnitude = self.min_pwm + self.feedforward_slope * abs(target_rad_s)
        return math.copysign(magnitude * self.scale, target_rad_s)

    def update(self, target_rad_s, measured_rad_s, dt):
        if abs(target_rad_s) < 1e-6:
            self.reset()
            return 0

        error = target_rad_s - measured_rad_s
        self.integral = clamp(
            self.integral + error * max(dt, 0.0),
            -self.integral_limit,
            self.integral_limit,
        )
        output = (
            self.feedforward(target_rad_s)
            + self.kp * error
            + self.ki * self.integral
        )
        output = clamp(output, -self.max_pwm, self.max_pwm)
        return int(round(output))


class PwmCalibrationCurve:
    def __init__(self, points):
        cleaned = sorted(
            (float(speed), float(pwm))
            for speed, pwm in points
            if float(speed) > 0.0 and float(pwm) > 0.0
        )
        if not cleaned:
            raise ValueError('calibration curve requires at least one point')
        self.points = cleaned

    def pwm_for_speed(self, target_rad_s):
        target = abs(float(target_rad_s))
        if len(self.points) == 1:
            return self.points[0][1]

        if target <= self.points[0][0]:
            first = self.points[0]
            second = self.points[1]
            return max(first[1], self._interpolate(target, first, second))

        for lower, upper in zip(self.points, self.points[1:]):
            if target <= upper[0]:
                return self._interpolate(target, lower, upper)

        return self._interpolate(
            target,
            self.points[-2],
            self.points[-1],
        )

    @staticmethod
    def _interpolate(target, lower, upper):
        speed_span = upper[0] - lower[0]
        if abs(speed_span) < 1e-9:
            return max(lower[1], upper[1])
        ratio = (target - lower[0]) / speed_span
        return lower[1] + ratio * (upper[1] - lower[1])


def load_pwm_calibration(path):
    calibration_path = Path(path).expanduser()
    groups = {}
    with calibration_path.open(newline='') as file:
        for row in csv.DictReader(file):
            label = row.get('label', '')
            try:
                pwm = float(row['pwm'])
                speed = float(row['command_rad_s_abs'])
            except (KeyError, TypeError, ValueError):
                continue
            groups.setdefault(label, []).append((speed, pwm))

    missing = [
        label for label in PWM_CALIBRATION_LABELS
        if not groups.get(label)
    ]
    if missing:
        raise ValueError(
            'calibration CSV is missing: ' + ', '.join(missing)
        )
    return {
        label: PwmCalibrationCurve(groups[label])
        for label in PWM_CALIBRATION_LABELS
    }


def summarize_pwm_calibration(path, min_points=1, min_duration_s=0.0):
    calibration_path = Path(path).expanduser()
    summary = {
        'path': str(calibration_path),
        'loaded': False,
        'ok': False,
        'error': '',
        'labels': {},
        'min_points_required': int(min_points),
        'min_duration_required_s': float(min_duration_s),
        'min_duration_s': None,
    }
    try:
        load_pwm_calibration(calibration_path)
    except (OSError, ValueError) as exc:
        summary['error'] = str(exc)
        return summary

    by_label = {label: [] for label in PWM_CALIBRATION_LABELS}
    try:
        with calibration_path.open(newline='') as file:
            for row in csv.DictReader(file):
                label = row.get('label', '')
                if label in by_label:
                    by_label[label].append(row)
    except OSError as exc:
        summary['error'] = str(exc)
        return summary

    summary['loaded'] = True
    durations = []
    errors = []
    for label in PWM_CALIBRATION_LABELS:
        rows = by_label[label]
        pwm_values = set()
        label_durations = []
        for row in rows:
            try:
                pwm_values.add(int(float(row['pwm'])))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                label_durations.append(float(row['duration_s']))
            except (KeyError, TypeError, ValueError):
                pass

        if label_durations:
            durations.extend(label_durations)
        min_label_duration = (
            min(label_durations) if label_durations else None
        )
        summary['labels'][label] = {
            'points': len(pwm_values),
            'min_duration_s': min_label_duration,
        }
        if len(pwm_values) < min_points:
            errors.append(
                f'{label} has {len(pwm_values)} PWM point(s); '
                f'need at least {min_points}'
            )
        if (
            min_label_duration is not None
            and min_label_duration < min_duration_s
        ):
            errors.append(
                f'{label} duration min {min_label_duration:.2f}s; '
                f'need >= {min_duration_s:.2f}s'
            )

    if durations:
        summary['min_duration_s'] = min(durations)
    if errors:
        summary['error'] = '; '.join(errors)
        return summary

    summary['ok'] = True
    return summary


class DifferentialOdometry:
    def __init__(
        self,
        wheel_radius_m,
        wheel_separation_m,
        encoder_cpr,
        left_encoder_sign=1,
        right_encoder_sign=-1,
    ):
        self.wheel_radius_m = float(wheel_radius_m)
        self.wheel_separation_m = float(wheel_separation_m)
        self.encoder_cpr = float(encoder_cpr)
        self.left_encoder_sign = int(left_encoder_sign)
        self.right_encoder_sign = int(right_encoder_sign)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_left_ticks = None
        self.last_right_ticks = None

    @property
    def meters_per_tick(self):
        return 2.0 * math.pi * self.wheel_radius_m / self.encoder_cpr

    def update(self, left_ticks, right_ticks):
        left_ticks *= self.left_encoder_sign
        right_ticks *= self.right_encoder_sign

        if self.last_left_ticks is None:
            self.last_left_ticks = left_ticks
            self.last_right_ticks = right_ticks
            return 0.0, 0.0

        delta_left = (
            left_ticks - self.last_left_ticks
        ) * self.meters_per_tick
        delta_right = (
            right_ticks - self.last_right_ticks
        ) * self.meters_per_tick
        self.last_left_ticks = left_ticks
        self.last_right_ticks = right_ticks

        delta_distance = 0.5 * (delta_left + delta_right)
        delta_yaw = (
            delta_right - delta_left
        ) / self.wheel_separation_m
        midpoint_yaw = self.yaw + 0.5 * delta_yaw
        self.x += delta_distance * math.cos(midpoint_yaw)
        self.y += delta_distance * math.sin(midpoint_yaw)
        self.yaw = normalize_angle(self.yaw + delta_yaw)
        return delta_distance, delta_yaw


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))
