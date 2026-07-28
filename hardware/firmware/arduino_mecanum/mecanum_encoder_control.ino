/*
  Four-wheel mecanum closed-loop controller for Arduino UNO.

  This replaces the open-loop mecanum_pwm_control sketch. It adds four
  quadrature encoders, per-wheel velocity PID (ported from
  encoder_pwm_test.ino), and a cascaded position mode with synchronized
  trapezoidal profiles for precise relative moves.

  Motor drivers: 2x Cytron MDD10A (PWM = speed, DIR = direction,
  sign-magnitude). Pin map (see docs/hardware/mecanum_wiring.md):
    Front-left  motor: D4 = DIR, D5 = PWM   (MDD10A #1 ch M1)
    Front-right motor: D7 = DIR, D6 = PWM   (MDD10A #1 ch M2)
    Rear-left   motor: D2 = DIR, D3 = PWM
    Rear-right  motor: D12 = DIR, D10 = PWM
    Front-left  encoder: A = D8, B = D9     (unchanged from diff base)
    Front-right encoder: A = A0, B = A1
    Rear-left   encoder: A = A2, B = A3
    Rear-right  encoder: A = A4, B = A5
    Spare: D11, D13. D0/D1 reserved for USB serial.

  All four motor PWM pins are hardware PWM (Timer0: D5/D6, Timer2: D3,
  Timer1: D10). Encoders are decoded with pin-change interrupts and a
  quadrature state table, so no external-interrupt pins are required.

  Serial protocol at 115200 baud (one command per line):
    m <vx_mps> <vy_mps> <wz_rad_s>
        Body velocity command (velocity mode). +x forward, +y left, +wz CCW.
    w <fl> <fr> <rl> <rr>
        Wheel angular velocity targets in rad/s (velocity mode).
    d <dx_m> <dy_m> <dyaw_rad>
        Relative body displacement (position mode). Runs a synchronized
        trapezoidal profile on all four wheels, replies "OK move", then
        "DONE,move,..." when settled or "ERR move timeout".
    pw <fl_rad> <fr_rad> <rl_rad> <rr_rad>
        Relative wheel position deltas in rad (position mode).
    p <fl_pwm> <fr_pwm> <rl_pwm> <rr_pwm>
        Direct signed PWM, -255..255 (disables closed loop until next m/w/d).
    pid <kp> <ki> <kd> <min_pwm> <max_pwm> <ff_slope> <integral_limit>
        Velocity PID + feedforward tuning (shared by all wheels).
    ppid <kp> <kd> <max_rad_s> <accel_rad_s2> <tol_rad> <hold_ms> <min_rad_s>
        Position loop tuning.
    geom <wheel_radius_m> <half_length_m> <half_width_m>
        Kinematics geometry used by m/d commands.
    sign <m_fl> <m_fr> <m_rl> <m_rr> <e_fl> <e_fr> <e_rl> <e_rr>
        Per-wheel motor output signs and encoder counting signs (+1/-1).
        Lets bring-up fix wiring polarity without reflashing.
    mstyle <fl> <fr> <rl> <rr>
        Per-wheel PWM encoding (0=DIR_HIGH_PWM_LOW, 1=PWM_HIGH_DIR_LOW).
        Switch driver boards without reflashing (old diff-drive driver =
        `mstyle 0 1 0 1`, MDD10A = `mstyle 1 1 1 1`).
    z       Zero encoder counters.
    stop    Stop all motors, abort any position move.
    stream <0|1>   Enable/disable periodic STATE lines.
    pins    Print pin map.
    test [pwm] [ms]   Drive each wheel forward/reverse in sequence.
    ?       Print help.

  STATE line format (10 Hz when streaming):
    STATE,<ms>,<fl_ticks>,<fr_ticks>,<rl_ticks>,<rr_ticks>,
          <fl_pwm>,<fr_pwm>,<rl_pwm>,<rr_pwm>,<mode>
  where <mode> is 0=idle/pwm, 1=velocity, 2=position.
*/

#include <Arduino.h>
#include <avr/interrupt.h>
#include <util/atomic.h>

enum WheelIndex : uint8_t {
  FL = 0,
  FR = 1,
  RL = 2,
  RR = 3,
  WHEEL_COUNT = 4
};

enum MotorStyle : uint8_t {
  DIR_HIGH_PWM_LOW = 0,  // forward: DIR HIGH, PWM = 255 - duty
  PWM_HIGH_DIR_LOW = 1   // forward: PWM = duty, DIR LOW
};

enum ControlMode : uint8_t {
  MODE_IDLE = 0,
  MODE_VELOCITY = 1,
  MODE_POSITION = 2
};

struct MotorConfig {
  const char *name;
  uint8_t dirPin;
  uint8_t pwmPin;
  MotorStyle style;
};

// Drivers: 2x Cytron MDD10A (2 channels each, sign-magnitude inputs:
// PWM duty = speed, DIR level = direction). ALL channels therefore use the
// PWM_HIGH_DIR_LOW style; the legacy DIR_HIGH_PWM_LOW (inverted-PWM) style
// only applied to the old driver and MUST NOT be used with the MDD10A.
// Per-wheel rotation direction is calibrated with the `sign` command.
const MotorConfig MOTORS[WHEEL_COUNT] = {
  {"front_left", 4, 5, PWM_HIGH_DIR_LOW},
  {"front_right", 7, 6, PWM_HIGH_DIR_LOW},
  {"rear_left", 2, 3, PWM_HIGH_DIR_LOW},
  {"rear_right", 12, 10, PWM_HIGH_DIR_LOW},
};

// Encoder pins. FL stays on the existing D8/D9 wiring (PCINT0, port B).
// FR/RL live on A0-A3 (PCINT1, port C). RR-A moved to D11 (2026-07-14: the
// A4 line read stuck-LOW on the bench, so RR quadrature is split across
// ports — A on PB3, B on PC5 — and decoded via the shared rrPrevState).
const uint8_t ENC_FL_A = 8;   // PB0
const uint8_t ENC_FL_B = 9;   // PB1
const uint8_t ENC_FR_A = A0;  // PC0
const uint8_t ENC_FR_B = A1;  // PC1
const uint8_t ENC_RL_A = A2;  // PC2
const uint8_t ENC_RL_B = A3;  // PC3
const uint8_t ENC_RR_A = 11;  // PB3 (was A4 — dead line)
const uint8_t ENC_RR_B = A5;  // PC5

const float TWO_PI_F = 6.28318530718f;
// JGB37-520 (12V 333rpm): 11 PPR hall x4 decode x1:30 gearbox = 1320.
// Bench-confirmed 2026-07-14 via no-load full-PWM tick rate (~1330 measured).
const float ENCODER_CPR = 1320.0f;
const unsigned long PID_INTERVAL_MS = 20;
const unsigned long REPORT_INTERVAL_MS = 100;
const unsigned long VELOCITY_TIMEOUT_MS = 500;
const unsigned long TEST_STOP_MS = 300;
const float TARGET_DEADBAND_RAD_S = 0.01f;

// Kinematics defaults; overridable with the geom command. Re-measure at the
// wheel contact centers once the mecanum frame is assembled.
// 2026-07-15 LiDAR 실측: 0.034 설정 시 직선 실이동 112~116% → 유효 반경 0.0388.
float wheelRadiusM = 0.0388f;
// 2026-07-15 롤러 X-config 재장착 후 gyro 재캘리브레이션: 유효 (L+W)=0.198
// (real.yaml과 동일, 물리 축간거리 0.150/0.125와 다른 유효값).
// 주의: 이 기본값은 아직 재플래시 안 됨 — 런타임은 브리지 geom push로 적용됨.
float baseHalfLengthM = 0.108f;
float baseHalfWidthM = 0.090f;

// Velocity PID + feedforward, shared across wheels (per-wheel trim happens on
// the ROS side through command scaling if ever needed).
// 2026-07-15 바닥(부하) 튜닝값을 기본값으로 채택 — 벤치 스크립트가 시리얼 오픈
// (UNO 리셋) 직후에도 튜닝 상태로 시작하도록. real.yaml과 동일해야 한다.
float velKp = 10.0f;
float velKi = 8.0f;
float velKd = 0.0f;
float velMinPwm = 10.0f;
float velMaxPwm = 250.0f;
float velFfSlope = 6.9f;
float velIntegralLimit = 20.0f;

// Position loop tuning.
float posKp = 6.0f;            // rad/s per rad of position error
float posKd = 0.0f;
float posMaxRadS = 6.0f;       // profile cruise velocity
float posAccelRadS2 = 12.0f;   // profile acceleration
// 2026-07-15: 0.035(~1mm)는 바닥 부하+RR 단채널 정착 노이즈로 hold 실패가
// 잦아 move timeout을 유발 → 바퀴 ~3mm에 해당하는 0.08로 완화 (real.yaml 동일).
float posToleranceRad = 0.08f;
unsigned long posHoldMs = 150;
// Floor so small errors still break stiction. 0.8은 FL 역방향 데드밴드에
// 못 미쳐 CCW 정착 타임아웃 유발 (2026-07-15) → 1.4.
float posMinRadS = 1.4f;

// 2026-07-14 bench calibration (MK4 mecanum + JGB37-520 + MDD10A): all four
// motors are wired +PWM=reverse, FR/RR encoders have A/B swapped. Baked in as
// defaults so bench scripts (which reset the UNO on serial open) start
// calibrated; the ROS bridge pushes the same values from real.yaml.
int8_t motorSign[WHEEL_COUNT] = {-1, -1, -1, -1};
int8_t encoderSign[WHEEL_COUNT] = {1, -1, 1, -1};

// Runtime-overridable PWM encoding style per wheel (0 = DIR_HIGH_PWM_LOW,
// 1 = PWM_HIGH_DIR_LOW). Defaults match the compile-time MOTORS[].style
// (MDD10A: all PWM_HIGH_DIR_LOW). The `mstyle` command lets bring-up switch
// to a legacy/old-driver board without reflashing, e.g. the diff-drive
// driver's left/right mixed convention `mstyle 0 1 0 1`.
MotorStyle motorStyle[WHEEL_COUNT] = {
  PWM_HIGH_DIR_LOW, PWM_HIGH_DIR_LOW, PWM_HIGH_DIR_LOW, PWM_HIGH_DIR_LOW
};

volatile long ticks[WHEEL_COUNT] = {0, 0, 0, 0};
volatile uint8_t prevPortB = 0;
volatile uint8_t prevPortC = 0;
// RR quadrature state (A<<1)|B spans two ports (A=PB3/D11, B=PC5/A5), so both
// pin-change ISRs update it through this shared variable.
volatile uint8_t rrPrevState = 0;
// 2026-07-14: the RR motor's encoder module has a dead A output (stuck LOW).
// `rr1ch 1` switches RR to single-channel counting: every B edge adds
// rrFbRawDir (+-2 so the effective CPR stays 1320), with the direction taken
// from the last commanded PWM sign. Loses accuracy only around direction
// reversals; disable again after the motor/encoder is replaced.
volatile bool rrSingleChannel = true;  // default ON until the encoder is replaced
volatile int8_t rrFbRawDir = 2;

ControlMode mode = MODE_IDLE;
int currentPwm[WHEEL_COUNT] = {0, 0, 0, 0};
float targetRadS[WHEEL_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
float measuredRadS[WHEEL_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
float velIntegral[WHEEL_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
float velLastError[WHEEL_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
long lastPidTicks[WHEEL_COUNT] = {0, 0, 0, 0};

// Position move state.
float moveStartRad[WHEEL_COUNT];
float moveDeltaRad[WHEEL_COUNT];
float movePeakRadS[WHEEL_COUNT];
float moveAccelRadS2[WHEEL_COUNT];
float moveLastPosError[WHEEL_COUNT];
float moveAccelTimeS = 0.0f;
float moveCruiseTimeS = 0.0f;
float moveTotalTimeS = 0.0f;
unsigned long moveStartMs = 0;
unsigned long moveDeadlineMs = 0;
unsigned long moveSettledSinceMs = 0;
bool moveSettling = false;

bool streamEnabled = true;
bool autoRunning = false;
bool abortAuto = false;
unsigned long lastVelocityCommandMs = 0;
unsigned long lastPidMs = 0;
unsigned long lastReportMs = 0;

char commandBuffer[160];
uint8_t commandLength = 0;

static inline int8_t quadratureDelta(uint8_t previousState, uint8_t currentState) {
  // State is encoded as (A << 1) | B.
  static const int8_t table[16] = {
     0,  1, -1,  0,
    -1,  0,  0,  1,
     1,  0,  0, -1,
     0, -1,  1,  0
  };
  return table[(previousState << 2) | currentState];
}

ISR(PCINT0_vect) {
  const uint8_t portB = PINB & (_BV(PB0) | _BV(PB1) | _BV(PB3));  // D8, D9, D11
  const uint8_t changed = portB ^ prevPortB;

  if (changed & (_BV(PB0) | _BV(PB1))) {
    const uint8_t prevState = ((prevPortB & _BV(PB0)) ? 2 : 0) | ((prevPortB & _BV(PB1)) ? 1 : 0);
    const uint8_t currState = ((portB & _BV(PB0)) ? 2 : 0) | ((portB & _BV(PB1)) ? 1 : 0);
    ticks[FL] += quadratureDelta(prevState, currState);
  }
  if (changed & _BV(PB3)) {  // RR-A toggled; B is read live from PC5
    const uint8_t currState = ((portB & _BV(PB3)) ? 2 : 0) | ((PINC & _BV(PC5)) ? 1 : 0);
    ticks[RR] += quadratureDelta(rrPrevState, currState);
    rrPrevState = currState;
  }
  prevPortB = portB;
}

ISR(PCINT1_vect) {
  const uint8_t portC = PINC & 0x3F;  // PC0-PC5 = A0-A5
  const uint8_t changed = portC ^ prevPortC;

  if (changed & (_BV(PC0) | _BV(PC1))) {
    const uint8_t prevState = ((prevPortC & _BV(PC0)) ? 2 : 0) | ((prevPortC & _BV(PC1)) ? 1 : 0);
    const uint8_t currState = ((portC & _BV(PC0)) ? 2 : 0) | ((portC & _BV(PC1)) ? 1 : 0);
    ticks[FR] += quadratureDelta(prevState, currState);
  }
  if (changed & (_BV(PC2) | _BV(PC3))) {
    const uint8_t prevState = ((prevPortC & _BV(PC2)) ? 2 : 0) | ((prevPortC & _BV(PC3)) ? 1 : 0);
    const uint8_t currState = ((portC & _BV(PC2)) ? 2 : 0) | ((portC & _BV(PC3)) ? 1 : 0);
    ticks[RL] += quadratureDelta(prevState, currState);
  }
  if (changed & _BV(PC5)) {  // RR-B toggled; A is read live from PB3 (D11)
    if (rrSingleChannel) {
      ticks[RR] += rrFbRawDir;
    } else {
      const uint8_t currState = ((PINB & _BV(PB3)) ? 2 : 0) | ((portC & _BV(PC5)) ? 1 : 0);
      ticks[RR] += quadratureDelta(rrPrevState, currState);
      rrPrevState = currState;
    }
  }
  prevPortC = portC;
}

long readTicks(uint8_t wheel) {
  long value;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    value = ticks[wheel];
  }
  return (long)encoderSign[wheel] * value;
}

float wheelPositionRad(uint8_t wheel) {
  return (float)readTicks(wheel) * TWO_PI_F / ENCODER_CPR;
}

void setMotorPwm(uint8_t wheel, int pwm) {
  if (wheel >= WHEEL_COUNT) {
    return;
  }
  const MotorConfig &motor = MOTORS[wheel];
  if (wheel == RR && pwm != 0) {
    // Single-channel fallback direction. `pwm` here is still the logical
    // command (positive = wheel physically forward); the encoderSign factor
    // keeps the raw count compatible with readTicks() like quadrature would.
    rrFbRawDir = (int8_t)((pwm > 0 ? 2 : -2) * encoderSign[RR]);
  }
  pwm = constrain(pwm * motorSign[wheel], -255, 255);
  currentPwm[wheel] = pwm;

  const uint8_t duty = abs(pwm);
  if (pwm == 0) {
    analogWrite(motor.pwmPin, 0);
    digitalWrite(motor.dirPin, LOW);
    return;
  }

  if (motorStyle[wheel] == DIR_HIGH_PWM_LOW) {
    if (pwm > 0) {
      digitalWrite(motor.dirPin, HIGH);
      analogWrite(motor.pwmPin, 255 - duty);
    } else {
      digitalWrite(motor.dirPin, LOW);
      analogWrite(motor.pwmPin, duty);
    }
    return;
  }

  // MDD10A sign-magnitude: PWM duty = |speed| in BOTH directions, DIR picks
  // the direction. (255 - duty here was a leftover from the legacy
  // inverted-PWM driver and made reverse duty complement the command:
  // full reverse produced duty 0.)
  if (pwm > 0) {
    digitalWrite(motor.dirPin, LOW);
    analogWrite(motor.pwmPin, duty);
  } else {
    digitalWrite(motor.dirPin, HIGH);
    analogWrite(motor.pwmPin, duty);
  }
}

void resetVelocityLoops() {
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    velIntegral[wheel] = 0.0f;
    velLastError[wheel] = 0.0f;
  }
}

void stopMotors() {
  mode = MODE_IDLE;
  moveSettling = false;
  resetVelocityLoops();
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    targetRadS[wheel] = 0.0f;
    setMotorPwm(wheel, 0);
  }
}

float rotationFactor() {
  return baseHalfLengthM + baseHalfWidthM;
}

void bodyToWheels(float vx, float vy, float wz, float out[WHEEL_COUNT]) {
  const float k = rotationFactor();
  out[FL] = (vx - vy - k * wz) / wheelRadiusM;
  out[FR] = (vx + vy + k * wz) / wheelRadiusM;
  out[RL] = (vx + vy - k * wz) / wheelRadiusM;
  out[RR] = (vx - vy + k * wz) / wheelRadiusM;
}

void scaleWheelTargets(float wheels[WHEEL_COUNT], float limit) {
  float peak = 0.0f;
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    peak = max(peak, abs(wheels[wheel]));
  }
  if (peak > limit && peak > 0.0f) {
    const float scale = limit / peak;
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      wheels[wheel] *= scale;
    }
  }
}

// 2026-07-17: 16→30 상향. 16은 r_eff 0.0388 기준 0.62m/s로 모터(무부하
// 34.9rad/s = 1.35m/s)보다 한참 낮은 인위적 캡이었음. 실효 상한은
// real.yaml max_wheel_rad_s(브리지 측 클램프)로 관리한다.
const float MAX_WHEEL_RAD_S = 30.0f;

void enterVelocityMode(const float wheels[WHEEL_COUNT]) {
  float clamped[WHEEL_COUNT];
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    clamped[wheel] = wheels[wheel];
  }
  scaleWheelTargets(clamped, MAX_WHEEL_RAD_S);
  if (mode != MODE_VELOCITY) {
    resetVelocityLoops();
  }
  mode = MODE_VELOCITY;
  moveSettling = false;
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    targetRadS[wheel] = clamped[wheel];
  }
  lastVelocityCommandMs = millis();
}

void startPositionMove(const float deltaRad[WHEEL_COUNT]) {
  float dMax = 0.0f;
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    dMax = max(dMax, abs(deltaRad[wheel]));
  }
  if (dMax < 1e-4f) {
    Serial.println(F("DONE,move,0,0,0,0"));
    return;
  }

  // Trapezoid on the longest wheel; other wheels scale proportionally so all
  // wheels start and finish together.
  float accelTime = posMaxRadS / posAccelRadS2;
  float accelDist = 0.5f * posAccelRadS2 * accelTime * accelTime;
  float peak = posMaxRadS;
  float cruiseTime;
  if (2.0f * accelDist >= dMax) {
    // Triangle profile.
    accelTime = sqrt(dMax / posAccelRadS2);
    peak = posAccelRadS2 * accelTime;
    cruiseTime = 0.0f;
  } else {
    cruiseTime = (dMax - 2.0f * accelDist) / posMaxRadS;
  }

  moveAccelTimeS = accelTime;
  moveCruiseTimeS = cruiseTime;
  moveTotalTimeS = 2.0f * accelTime + cruiseTime;

  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    const float ratio = deltaRad[wheel] / dMax;  // signed
    moveStartRad[wheel] = wheelPositionRad(wheel);
    moveDeltaRad[wheel] = deltaRad[wheel];
    movePeakRadS[wheel] = peak * ratio;
    moveAccelRadS2[wheel] = posAccelRadS2 * ratio;
    moveLastPosError[wheel] = 0.0f;
  }

  resetVelocityLoops();
  mode = MODE_POSITION;
  moveSettling = false;
  moveStartMs = millis();
  // 2026-07-15: 정착 여유 1.5s는 바닥 부하에서 빠듯해 timeout 오탐 → 2.5s.
  // 2026-07-19: 접근 단발이동(0.3~0.4m)이 경기장에서 timeout → 파지 허공 CLOSE.
  //   여유 2.5→5.0s (스톨 보호는 유지하되 오탐 억제).
  moveDeadlineMs = moveStartMs
    + (unsigned long)(moveTotalTimeS * 1000.0f) * 2UL
    + 5000UL;
}

// Profile position/velocity for one wheel at elapsed time t.
void sampleProfile(uint8_t wheel, float t, float *posRad, float *velRadS) {
  const float a = moveAccelRadS2[wheel];
  const float vp = movePeakRadS[wheel];
  const float ta = moveAccelTimeS;
  const float tc = moveCruiseTimeS;

  if (t <= 0.0f) {
    *posRad = 0.0f;
    *velRadS = 0.0f;
  } else if (t < ta) {
    *posRad = 0.5f * a * t * t;
    *velRadS = a * t;
  } else if (t < ta + tc) {
    *posRad = 0.5f * a * ta * ta + vp * (t - ta);
    *velRadS = vp;
  } else if (t < moveTotalTimeS) {
    const float td = t - ta - tc;
    *posRad = 0.5f * a * ta * ta + vp * tc + vp * td - 0.5f * a * td * td;
    *velRadS = vp - a * td;
  } else {
    *posRad = moveDeltaRad[wheel];
    *velRadS = 0.0f;
  }
}

int velocityPidStep(uint8_t wheel, float target, float measured, float dt) {
  if (abs(target) < TARGET_DEADBAND_RAD_S) {
    velIntegral[wheel] = 0.0f;
    velLastError[wheel] = 0.0f;
    return 0;
  }
  const float error = target - measured;
  velIntegral[wheel] = constrain(
    velIntegral[wheel] + error * dt,
    -velIntegralLimit,
    velIntegralLimit
  );
  const float derivative = dt > 0.0f ? (error - velLastError[wheel]) / dt : 0.0f;
  velLastError[wheel] = error;

  const float feedforward = velMinPwm + velFfSlope * abs(target);
  float output = feedforward * (target >= 0.0f ? 1.0f : -1.0f)
    + velKp * error
    + velKi * velIntegral[wheel]
    + velKd * derivative;
  output = constrain(output, -velMaxPwm, velMaxPwm);
  return (int)round(output);
}

void finishPositionMove(bool timedOut) {
  stopMotors();
  if (timedOut) {
    Serial.println(F("ERR move timeout"));
    return;
  }
  Serial.print(F("DONE,move"));
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    const float err = moveStartRad[wheel] + moveDeltaRad[wheel]
      - wheelPositionRad(wheel);
    Serial.print(',');
    Serial.print(err, 4);
  }
  Serial.println();
}

void runControlStep() {
  const unsigned long now = millis();
  if (now - lastPidMs < PID_INTERVAL_MS) {
    return;
  }
  const float dt = (now - lastPidMs) / 1000.0f;
  lastPidMs = now;

  // Measure wheel velocities from tick deltas.
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    const long t = readTicks(wheel);
    measuredRadS[wheel] = (float)(t - lastPidTicks[wheel]) * TWO_PI_F
      / ENCODER_CPR / dt;
    lastPidTicks[wheel] = t;
  }

  if (mode == MODE_VELOCITY) {
    if (now - lastVelocityCommandMs > VELOCITY_TIMEOUT_MS) {
      stopMotors();
      Serial.println(F("WARN velocity timeout"));
      return;
    }
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      setMotorPwm(
        wheel,
        velocityPidStep(wheel, targetRadS[wheel], measuredRadS[wheel], dt)
      );
    }
    return;
  }

  if (mode == MODE_POSITION) {
    if (now > moveDeadlineMs) {
      finishPositionMove(true);
      return;
    }
    const float t = (now - moveStartMs) / 1000.0f;
    bool allSettled = true;
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      float profilePos;
      float profileVel;
      sampleProfile(wheel, t, &profilePos, &profileVel);
      const float actual = wheelPositionRad(wheel) - moveStartRad[wheel];
      const float posError = profilePos - actual;
      const float dError = dt > 0.0f
        ? (posError - moveLastPosError[wheel]) / dt
        : 0.0f;
      moveLastPosError[wheel] = posError;

      float velTarget = profileVel + posKp * posError + posKd * dError;

      const float finalError = moveDeltaRad[wheel] - actual;
      const bool profileDone = t >= moveTotalTimeS;
      if (profileDone) {
        if (abs(finalError) > posToleranceRad) {
          allSettled = false;
          // Floor the correction speed so stiction cannot stall the wheel
          // just below the PWM deadband (the diff-drive final-yaw stall).
          if (abs(velTarget) < posMinRadS) {
            velTarget = finalError >= 0.0f ? posMinRadS : -posMinRadS;
          }
        } else {
          velTarget = 0.0f;
        }
      } else {
        allSettled = false;
      }

      // 2026-07-17: MAX_WHEEL_RAD_S(30)가 아니라 이 move의 크루즈 캡으로
      // 클램프. posKp 보정이 무제한 가산되면 max_v를 넘는 폭주 →
      // 종단 급감속/슬립/물체 타격 (16→30 상향 후 실제 발생).
      // 여유 1.15배: 정상 추적 보정은 허용, 폭주만 차단.
      const float vCap = posMaxRadS * 1.15f;
      velTarget = constrain(velTarget, -vCap, vCap);
      targetRadS[wheel] = velTarget;
      setMotorPwm(
        wheel,
        velocityPidStep(wheel, velTarget, measuredRadS[wheel], dt)
      );
    }

    if (allSettled) {
      if (!moveSettling) {
        moveSettling = true;
        moveSettledSinceMs = now;
      } else if (now - moveSettledSinceMs >= posHoldMs) {
        finishPositionMove(false);
      }
    } else {
      moveSettling = false;
    }
  }
}

void printHelp() {
  Serial.println(F("READY mecanum_encoder_control"));
  Serial.println(F("Commands: m <vx> <vy> <wz>, w <fl> <fr> <rl> <rr>, d <dx_m> <dy_m> <dyaw_rad>, pw <fl> <fr> <rl> <rr>, p <fl> <fr> <rl> <rr>, pid <kp> <ki> <kd> <min_pwm> <max_pwm> <ff_slope> <int_limit>, ppid <kp> <kd> <max_rad_s> <accel> <tol_rad> <hold_ms> <min_rad_s>, geom <r> <half_l> <half_w>, sign <m_fl> <m_fr> <m_rl> <m_rr> <e_fl> <e_fr> <e_rl> <e_rr>, mstyle <fl> <fr> <rl> <rr>, z, stop, stream <0|1>, pins, test [pwm] [ms], ?"));
  Serial.println(F("Axes: +x forward, +y left, +wz CCW."));
}

void printPins() {
  Serial.println(F("PINS front_left dir=D4 pwm=D5 encA=D8 encB=D9"));
  Serial.println(F("PINS front_right dir=D7 pwm=D6 encA=A0 encB=A1"));
  Serial.println(F("PINS rear_left dir=D2 pwm=D3 encA=A2 encB=A3"));
  Serial.println(F("PINS rear_right dir=D12 pwm=D10 encA=D11 encB=A5"));
  Serial.println(F("PINS serial_reserved=D0,D1 spare=D13 dead=A4"));
  Serial.print(F("MSTYLE "));
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    Serial.print((int)motorStyle[wheel]);
    Serial.print(wheel < WHEEL_COUNT - 1 ? ' ' : '\n');
  }
}

void printStatusLine() {
  Serial.print(F("STATE,"));
  Serial.print(millis());
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    Serial.print(',');
    Serial.print(readTicks(wheel));
  }
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    Serial.print(',');
    Serial.print(currentPwm[wheel]);
  }
  Serial.print(',');
  Serial.println((int)mode);
}

void maybePrintStatusLine() {
  if (!streamEnabled) {
    return;
  }
  const unsigned long now = millis();
  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = now;
    printStatusLine();
  }
}

void handleCommand(char *line);

void processSerialInput() {
  while (Serial.available() > 0) {
    const char c = Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      commandBuffer[commandLength] = '\0';
      handleCommand(commandBuffer);
      commandLength = 0;
      continue;
    }
    if (commandLength < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLength++] = c;
    }
  }
}

void runTimedWheelStep(const char *name, uint8_t wheel, int pwm, unsigned long durationMs) {
  if (abortAuto) {
    return;
  }
  Serial.print(F("STEP,"));
  Serial.print(name);
  Serial.print(',');
  Serial.println(pwm);

  const long startTicks = readTicks(wheel);
  setMotorPwm(wheel, pwm);
  const unsigned long startMs = millis();
  while (!abortAuto && millis() - startMs < durationMs) {
    processSerialInput();
    maybePrintStatusLine();
  }
  setMotorPwm(wheel, 0);
  Serial.print(F("STEP_TICKS,"));
  Serial.print(name);
  Serial.print(',');
  Serial.println(readTicks(wheel) - startTicks);
}

void runAutoTest(int pwm, unsigned long durationMs) {
  if (autoRunning) {
    Serial.println(F("ERR test already running"));
    return;
  }
  pwm = constrain(abs(pwm), 0, 255);
  if (pwm == 0) {
    pwm = 100;
  }
  if (durationMs < 200) {
    durationMs = 200;
  }

  autoRunning = true;
  abortAuto = false;
  mode = MODE_IDLE;
  Serial.println(F("OK test start"));

  const char *names[WHEEL_COUNT] = {
    "front_left", "front_right", "rear_left", "rear_right"
  };
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT && !abortAuto; wheel++) {
    char label[32];
    snprintf(label, sizeof(label), "%s_forward", names[wheel]);
    runTimedWheelStep(label, wheel, pwm, durationMs);
    delay(TEST_STOP_MS);
    snprintf(label, sizeof(label), "%s_reverse", names[wheel]);
    runTimedWheelStep(label, wheel, -pwm, durationMs);
    delay(TEST_STOP_MS);
  }
  stopMotors();
  Serial.println(abortAuto ? F("OK test aborted") : F("OK test done"));
  abortAuto = false;
  autoRunning = false;
}

bool parseFloats(float *out, uint8_t count) {
  for (uint8_t index = 0; index < count; index++) {
    char *arg = strtok(NULL, " ,\t");
    if (arg == NULL) {
      return false;
    }
    out[index] = atof(arg);
  }
  return true;
}

void handleCommand(char *line) {
  char *command = strtok(line, " ,\t");
  if (command == NULL) {
    return;
  }

  if (strcmp(command, "?") == 0 || strcmp(command, "help") == 0) {
    printHelp();
    return;
  }

  if (strcmp(command, "m") == 0 || strcmp(command, "move") == 0) {
    float args[3];
    if (!parseFloats(args, 3)) {
      Serial.println(F("ERR usage: m <vx_mps> <vy_mps> <wz_rad_s>"));
      return;
    }
    float wheels[WHEEL_COUNT];
    bodyToWheels(args[0], args[1], args[2], wheels);
    enterVelocityMode(wheels);
    Serial.println(F("OK m"));
    return;
  }

  if (strcmp(command, "w") == 0 || strcmp(command, "wheel") == 0) {
    float wheels[WHEEL_COUNT];
    if (!parseFloats(wheels, 4)) {
      Serial.println(F("ERR usage: w <fl> <fr> <rl> <rr>"));
      return;
    }
    enterVelocityMode(wheels);
    Serial.println(F("OK w"));
    return;
  }

  if (strcmp(command, "d") == 0 || strcmp(command, "disp") == 0) {
    float args[3];
    if (!parseFloats(args, 3)) {
      Serial.println(F("ERR usage: d <dx_m> <dy_m> <dyaw_rad>"));
      return;
    }
    float wheels[WHEEL_COUNT];
    // Displacement uses the same inverse kinematics as velocity: a body
    // displacement (dx, dy, dyaw) maps to wheel angle deltas.
    bodyToWheels(args[0], args[1], args[2], wheels);
    startPositionMove(wheels);
    Serial.println(F("OK move"));
    return;
  }

  if (strcmp(command, "pw") == 0) {
    float wheels[WHEEL_COUNT];
    if (!parseFloats(wheels, 4)) {
      Serial.println(F("ERR usage: pw <fl_rad> <fr_rad> <rl_rad> <rr_rad>"));
      return;
    }
    startPositionMove(wheels);
    Serial.println(F("OK move"));
    return;
  }

  if (strcmp(command, "p") == 0 || strcmp(command, "pwm") == 0) {
    float pwms[WHEEL_COUNT];
    if (!parseFloats(pwms, 4)) {
      Serial.println(F("ERR usage: p <fl> <fr> <rl> <rr>"));
      return;
    }
    mode = MODE_IDLE;
    moveSettling = false;
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      targetRadS[wheel] = 0.0f;
      setMotorPwm(wheel, (int)pwms[wheel]);
    }
    lastVelocityCommandMs = millis();
    Serial.println(F("OK pwm"));
    return;
  }

  if (strcmp(command, "pid") == 0) {
    float args[7];
    if (!parseFloats(args, 7)) {
      Serial.println(F("ERR usage: pid <kp> <ki> <kd> <min_pwm> <max_pwm> <ff_slope> <int_limit>"));
      return;
    }
    velKp = args[0];
    velKi = args[1];
    velKd = args[2];
    velMinPwm = constrain(args[3], 0.0f, 255.0f);
    velMaxPwm = constrain(args[4], velMinPwm, 255.0f);
    velFfSlope = max(0.0f, args[5]);
    velIntegralLimit = max(0.0f, args[6]);
    resetVelocityLoops();
    Serial.println(F("OK pid"));
    return;
  }

  if (strcmp(command, "ppid") == 0) {
    float args[7];
    if (!parseFloats(args, 7)) {
      Serial.println(F("ERR usage: ppid <kp> <kd> <max_rad_s> <accel> <tol_rad> <hold_ms> <min_rad_s>"));
      return;
    }
    posKp = max(0.0f, args[0]);
    posKd = max(0.0f, args[1]);
    posMaxRadS = constrain(args[2], 0.1f, MAX_WHEEL_RAD_S);
    posAccelRadS2 = max(0.1f, args[3]);
    posToleranceRad = max(0.001f, args[4]);
    posHoldMs = (unsigned long)max(0.0f, args[5]);
    posMinRadS = constrain(args[6], 0.0f, posMaxRadS);
    Serial.println(F("OK ppid"));
    return;
  }

  if (strcmp(command, "geom") == 0) {
    float args[3];
    if (!parseFloats(args, 3)) {
      Serial.println(F("ERR usage: geom <wheel_radius_m> <half_length_m> <half_width_m>"));
      return;
    }
    if (args[0] <= 0.0f || args[1] <= 0.0f || args[2] <= 0.0f) {
      Serial.println(F("ERR geom values must be positive"));
      return;
    }
    wheelRadiusM = args[0];
    baseHalfLengthM = args[1];
    baseHalfWidthM = args[2];
    Serial.println(F("OK geom"));
    return;
  }

  if (strcmp(command, "sign") == 0) {
    float args[8];
    if (!parseFloats(args, 8)) {
      Serial.println(F("ERR usage: sign <m_fl> <m_fr> <m_rl> <m_rr> <e_fl> <e_fr> <e_rl> <e_rr>"));
      return;
    }
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      motorSign[wheel] = args[wheel] >= 0.0f ? 1 : -1;
      encoderSign[wheel] = args[wheel + 4] >= 0.0f ? 1 : -1;
    }
    Serial.println(F("OK sign"));
    return;
  }

  if (strcmp(command, "rr1ch") == 0) {
    float arg;
    if (!parseFloats(&arg, 1)) {
      Serial.println(F("ERR usage: rr1ch <0|1>"));
      return;
    }
    rrSingleChannel = arg >= 0.5f;
    Serial.println(rrSingleChannel ? F("OK rr1ch on") : F("OK rr1ch off"));
    return;
  }

  if (strcmp(command, "mstyle") == 0) {
    float args[4];
    if (!parseFloats(args, 4)) {
      Serial.println(F("ERR usage: mstyle <fl> <fr> <rl> <rr>  (0=DIR_HIGH_PWM_LOW, 1=PWM_HIGH_DIR_LOW)"));
      return;
    }
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      motorStyle[wheel] = args[wheel] >= 0.5f ? PWM_HIGH_DIR_LOW : DIR_HIGH_PWM_LOW;
    }
    stopMotors();  // re-latch outputs under the new encoding
    Serial.print(F("OK mstyle "));
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      Serial.print((int)motorStyle[wheel]);
      Serial.print(wheel < WHEEL_COUNT - 1 ? ' ' : '\n');
    }
    return;
  }

  if (strcmp(command, "z") == 0 || strcmp(command, "zero") == 0) {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
        ticks[wheel] = 0;
      }
    }
    for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
      lastPidTicks[wheel] = 0;
    }
    Serial.println(F("OK zero"));
    return;
  }

  if (strcmp(command, "stop") == 0 || strcmp(command, "s") == 0) {
    stopMotors();
    abortAuto = true;
    Serial.println(F("OK stop"));
    return;
  }

  if (strcmp(command, "stream") == 0) {
    char *enabledArg = strtok(NULL, " ,\t");
    if (enabledArg == NULL) {
      Serial.println(F("ERR usage: stream <0|1>"));
      return;
    }
    streamEnabled = atoi(enabledArg) != 0;
    Serial.println(streamEnabled ? F("OK stream 1") : F("OK stream 0"));
    return;
  }

  if (strcmp(command, "pins") == 0) {
    printPins();
    return;
  }

  if (strcmp(command, "test") == 0 || strcmp(command, "auto") == 0) {
    char *pwmArg = strtok(NULL, " ,\t");
    char *msArg = strtok(NULL, " ,\t");
    const int testPwm = pwmArg == NULL ? 100 : atoi(pwmArg);
    const unsigned long testMs =
      msArg == NULL ? 800UL : strtoul(msArg, NULL, 10);
    runAutoTest(testPwm, testMs);
    return;
  }

  Serial.println(F("ERR unknown command"));
}

void setupEncoderInterrupts() {
  pinMode(ENC_FL_A, INPUT_PULLUP);
  pinMode(ENC_FL_B, INPUT_PULLUP);
  pinMode(ENC_FR_A, INPUT_PULLUP);
  pinMode(ENC_FR_B, INPUT_PULLUP);
  pinMode(ENC_RL_A, INPUT_PULLUP);
  pinMode(ENC_RL_B, INPUT_PULLUP);
  pinMode(ENC_RR_A, INPUT_PULLUP);
  pinMode(ENC_RR_B, INPUT_PULLUP);

  prevPortB = PINB & (_BV(PB0) | _BV(PB1) | _BV(PB3));
  prevPortC = PINC & 0x3F;
  rrPrevState = ((PINB & _BV(PB3)) ? 2 : 0) | ((PINC & _BV(PC5)) ? 1 : 0);

  PCICR |= _BV(PCIE0) | _BV(PCIE1);
  PCMSK0 = _BV(PCINT0) | _BV(PCINT1) | _BV(PCINT3);  // D8, D9, D11
  PCMSK1 = _BV(PCINT8) | _BV(PCINT9) | _BV(PCINT10)
    | _BV(PCINT11) | _BV(PCINT13);  // A0-A3, A5 (A4 dead — masked out)
}

void setup() {
  for (uint8_t wheel = 0; wheel < WHEEL_COUNT; wheel++) {
    pinMode(MOTORS[wheel].dirPin, OUTPUT);
    pinMode(MOTORS[wheel].pwmPin, OUTPUT);
  }
  setupEncoderInterrupts();
  stopMotors();
  Serial.begin(115200);
  delay(500);
  printHelp();
  printPins();
  lastPidMs = millis();
}

void loop() {
  processSerialInput();
  runControlStep();
  maybePrintStatusLine();
}
