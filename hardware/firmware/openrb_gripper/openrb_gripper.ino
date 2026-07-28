#include <Dynamixel2Arduino.h>

#define DXL_SERIAL Serial1
#define PC_SERIAL Serial

const int DXL_DIR_PIN = -1;

uint8_t DXL_ID = 0;
float DXL_PROTOCOL_VERSION = 2.0;
uint32_t DXL_BAUD = 1000000;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

using namespace ControlTableItem;

// 2026-07-16 그리퍼 재조립 후 실측 재캘리브레이션 (DYNAMIXEL Wizard):
// 닫힘 36.74도, 완전 열림 214.37도.
const float CLOSED_DEG = 36.74;
const float FULL_OPEN_DEG = 214.37;

const int DEFAULT_OPEN_CURRENT_RAW = 120;
const int DEFAULT_GRIP_CURRENT_RAW = 120;
const int MIN_GOAL_CURRENT_RAW = 10;
const int MAX_GOAL_CURRENT_RAW = 120;
const int CURRENT_FAULT_RAW = 140;
const unsigned long CURRENT_FAULT_HOLD_MS = 250;
const bool REQUIRE_CURRENT_BASED_POSITION = true;
const bool ENABLE_DXL_POWER_ON_BOOT = false;

int OPEN_CURRENT_RAW = DEFAULT_OPEN_CURRENT_RAW;
int GRIP_CURRENT_RAW = DEFAULT_GRIP_CURRENT_RAW;
int PROFILE_VELOCITY_RAW = 40;
int PROFILE_ACCELERATION_RAW = 10;
// 2026-07-17 확인: current-based position 모드에서 닫힘 속도는 goal current
// (토크)로 제한되어 profile velocity/accel 상향(150/40 실험)이 곡선에 전혀
// 영향 없었음 → 원복. 닫힘 가속하려면 전류 상한↑(파지력 트레이드오프) 필요.

bool dxl_ready = false;
bool current_based_mode = false;
bool fault_latched = false;
bool dxl_power_enabled = false;
int32_t actual_current_limit_raw = -1;
unsigned long current_fault_start_time = 0;

unsigned long last_cmd_time = 0;
const unsigned long WATCHDOG_MS = 3000;

// 카메라 마스트 리프트 (2026-07-16): 같은 버스의 2번째 XC330, ID 12.
// 그리퍼와 달리 멀티턴 스트로크(약 7.7회전 = 31641 ticks)라 상대 틱 이동만 지원.
// 워치독/OPEN/CLOSE는 리프트를 절대 건드리지 않는다.
//
// 홈 로직: 부팅 후 DXL 전원 인가 → 첫 리프트 초기화 시점의 위치를 "바닥 홈(0)"으로
// 간주해 1회 캡처한다(전원 유지 동안 멀티턴 카운터가 연속 추적하므로 유효). 최고점은
// 홈에서 LIFT_STROKE_TICKS 만큼 "감소 방향"(LIFT_RAISE_SIGN)으로 올린 지점이다.
// 실측(2026-07-17): 바닥 POS_RAW 33122 ↔ 최고점 1481 → 델타 31641, 올림=tick 감소.
const uint8_t LIFT_DXL_ID = 12;
const int LIFT_GOAL_CURRENT_RAW = 600;      // XC330 1mA/LSB, 스톨(~1.5A) 대비 여유
const int LIFT_PROFILE_VELOCITY_RAW = 400;  // 0.229rpm/LSB → 약 92rpm (velocity limit 근처). 전류 여유 크므로 상향
const int LIFT_PROFILE_ACCEL_RAW = 60;
const long LIFT_MAX_MOVE_TICKS = 40000;     // 오타 방지 상한 (풀스트로크 31641 < 40000)
const long LIFT_STROKE_TICKS = 31641;       // 실측 바닥↔최고 스트로크(하드 끝단, 마진 0)
const long LIFT_TOP_MARGIN_TICKS = 800;     // 최고점 하드 스톱 여유(≈0.2회전). LIFT_TO_TOP은 이만큼 못 미쳐 정지
const int  LIFT_RAISE_SIGN = -1;            // 올리는 방향: tick 감소

bool lift_ready = false;
bool lift_home_set = false;                 // DXL 전원 세션당 1회 캡처
int32_t lift_home_raw = 0;                  // 바닥 홈(부팅 시 위치) 절대 tick

const uint32_t SCAN_BAUDS[] = {
  57600,
  115200,
  1000000,
  2000000,
  3000000,
};
const uint8_t DEFAULT_SCAN_MAX_ID = 20;

float normalizeDeg(float deg) {
  while (deg < 0.0) {
    deg += 360.0;
  }

  while (deg >= 360.0) {
    deg -= 360.0;
  }

  return deg;
}

float shortestDeltaDeg(float from_deg, float to_deg) {
  float delta = normalizeDeg(to_deg) - normalizeDeg(from_deg);

  while (delta > 180.0) {
    delta -= 360.0;
  }

  while (delta <= -180.0) {
    delta += 360.0;
  }

  return delta;
}

float safeTravelDeg() {
  return FULL_OPEN_DEG - CLOSED_DEG;
}

float safeArcRatio(float deg) {
  float travel = safeTravelDeg();

  if (abs(travel) < 0.001) {
    return 0.0;
  }

  return (normalizeDeg(deg) - CLOSED_DEG) / travel;
}

bool safeArcWrapsZero() {
  return false;
}

bool degIsOnSafeArc(float deg) {
  float ratio = safeArcRatio(deg);
  return ratio >= 0.0 && ratio <= 1.0;
}

float ratioToDeg(float ratio) {
  ratio = constrain(ratio, 0.0, 1.0);
  return CLOSED_DEG + safeTravelDeg() * ratio;
}

float clampDeg(float deg) {
  float normalized_deg = normalizeDeg(deg);
  float travel = safeTravelDeg();

  if (degIsOnSafeArc(normalized_deg)) {
    return normalized_deg;
  }

  if (travel >= 0.0) {
    if (normalized_deg < CLOSED_DEG) {
      return CLOSED_DEG;
    }
    return FULL_OPEN_DEG;
  }

  if (normalized_deg > CLOSED_DEG) {
    return CLOSED_DEG;
  }
  return FULL_OPEN_DEG;
}

int degToRaw(float deg) {
  float safe_deg = clampDeg(deg);
  return (int)((safe_deg / 360.0) * 4095.0);
}

int clampCurrentRaw(int current_raw) {
  return constrain(current_raw, MIN_GOAL_CURRENT_RAW, MAX_GOAL_CURRENT_RAW);
}

float targetExtendedDeg(float safe_deg) {
  float present_deg = dxl.getPresentPosition(DXL_ID, UNIT_DEGREE);
  float present_ratio = constrain(safeArcRatio(present_deg), 0.0, 1.0);
  float target_ratio = constrain(safeArcRatio(safe_deg), 0.0, 1.0);
  float delta_deg = (target_ratio - present_ratio) * safeTravelDeg();
  return present_deg + delta_deg;
}

bool isUnsignedIntegerText(String text) {
  text.trim();

  if (text.length() == 0) {
    return false;
  }

  for (size_t index = 0; index < text.length(); index++) {
    char value = text.charAt(index);

    if (value < '0' || value > '9') {
      return false;
    }
  }

  return true;
}

void setDxlPower(bool enabled) {
#if defined(BDPIN_DXL_PWR_EN)
  pinMode(BDPIN_DXL_PWR_EN, OUTPUT);
  digitalWrite(BDPIN_DXL_PWR_EN, enabled ? HIGH : LOW);
  delay(300);
#endif
  dxl_power_enabled = enabled;
}

void enableDxlPower() {
  if (ENABLE_DXL_POWER_ON_BOOT) {
    setDxlPower(true);
  }
}

void restoreDxlPort() {
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
}

void printDxlConfig() {
  PC_SERIAL.print(" ID ");
  PC_SERIAL.print(DXL_ID);
  PC_SERIAL.print(" BAUD ");
  PC_SERIAL.print(DXL_BAUD);
  PC_SERIAL.print(" PROTOCOL ");
  PC_SERIAL.print(DXL_PROTOCOL_VERSION);
}

void moveToDeg(float target_deg, int current_raw) {
  float safe_deg = clampDeg(target_deg);
  int safe_current_raw = clampCurrentRaw(current_raw);

  if (!dxl_ready || fault_latched) {
    PC_SERIAL.print("ERR_DYNAMIXEL_NOT_READY");
    printDxlConfig();
    PC_SERIAL.println();
    return;
  }

  if (current_based_mode) {
    dxl.setGoalCurrent(DXL_ID, safe_current_raw);
  }

  float target_extended_deg = targetExtendedDeg(safe_deg);
  bool ok = dxl.setGoalPosition(DXL_ID, target_extended_deg, UNIT_DEGREE);

  if (ok) {
    PC_SERIAL.print("OK_MOVE_DEG ");
  } else {
    PC_SERIAL.print("ERR_MOVE_DEG ");
  }

  PC_SERIAL.print(safe_deg);
  PC_SERIAL.print(" TARGET_DEG ");
  PC_SERIAL.print(target_extended_deg);
  PC_SERIAL.print(" RAW ");
  PC_SERIAL.print(degToRaw(safe_deg));
  PC_SERIAL.print(" CURRENT_RAW ");
  PC_SERIAL.println(safe_current_raw);
}

void moveToDeg(float target_deg) {
  moveToDeg(target_deg, GRIP_CURRENT_RAW);
}

void moveToRatio(float ratio) {
  moveToDeg(ratioToDeg(ratio), GRIP_CURRENT_RAW);
}

void openGripper() {
  moveToDeg(FULL_OPEN_DEG, OPEN_CURRENT_RAW);
}

void closeGripper() {
  moveToDeg(CLOSED_DEG);
}

void printStatus() {
  PC_SERIAL.print("STATUS READY ");
  PC_SERIAL.print(dxl_ready ? 1 : 0);

  if (dxl_ready) {
    float pos_deg = dxl.getPresentPosition(DXL_ID, UNIT_DEGREE);
    int32_t pos_raw = (int32_t)dxl.getPresentPosition(DXL_ID);
    int32_t cur_raw = (int32_t)dxl.getPresentCurrent(DXL_ID);
    int32_t volt_raw = dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, DXL_ID);
    int32_t err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, DXL_ID);
    bool torque_enabled = dxl.getTorqueEnableStat(DXL_ID);
    float safe_ratio = safeArcRatio(pos_deg);

    PC_SERIAL.print(" POS_RAW ");
    PC_SERIAL.print(pos_raw);

    PC_SERIAL.print(" POS_DEG ");
    PC_SERIAL.print(pos_deg);

    PC_SERIAL.print(" POS_RATIO ");
    PC_SERIAL.print(constrain(safe_ratio, 0.0, 1.0));

    PC_SERIAL.print(" CURRENT_RAW ");
    PC_SERIAL.print(cur_raw);

    PC_SERIAL.print(" VOLTAGE_RAW ");
    PC_SERIAL.print(volt_raw);

    PC_SERIAL.print(" HW_ERROR ");
    PC_SERIAL.print(err);

    PC_SERIAL.print(" TORQUE ");
    PC_SERIAL.print(torque_enabled ? 1 : 0);
  }

  PC_SERIAL.print(" FAULT ");
  PC_SERIAL.print(fault_latched ? 1 : 0);
  PC_SERIAL.print(" CURRENT_LIMIT_RAW ");
  PC_SERIAL.print(MAX_GOAL_CURRENT_RAW);
  PC_SERIAL.print(" ACTUAL_CURRENT_LIMIT_RAW ");
  PC_SERIAL.print(actual_current_limit_raw);
  PC_SERIAL.print(" SAFE_ARC_WRAP ");
  PC_SERIAL.print(safeArcWrapsZero() ? 1 : 0);
  PC_SERIAL.print(" DXL_POWER ");
  PC_SERIAL.print(dxl_power_enabled ? 1 : 0);
  PC_SERIAL.print(" AUTO_TORQUE_OFF_MS 0");
  printDxlConfig();
  PC_SERIAL.println();
}

void checkCurrentFault() {
  if (!dxl_ready || fault_latched || !current_based_mode) {
    current_fault_start_time = 0;
    return;
  }

  int32_t err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, DXL_ID);
  if (err != 0) {
    dxl.torqueOff(DXL_ID);
    fault_latched = true;
    dxl_ready = false;
    PC_SERIAL.print("FAULT_HARDWARE_ERROR HW_ERROR ");
    PC_SERIAL.println(err);
    return;
  }

  int32_t cur_raw = (int32_t)dxl.getPresentCurrent(DXL_ID);

  if (abs(cur_raw) < CURRENT_FAULT_RAW) {
    current_fault_start_time = 0;
    return;
  }

  if (current_fault_start_time == 0) {
    current_fault_start_time = millis();
    return;
  }

  if (millis() - current_fault_start_time < CURRENT_FAULT_HOLD_MS) {
    return;
  }

  dxl.torqueOff(DXL_ID);
  fault_latched = true;
  dxl_ready = false;
  PC_SERIAL.print("FAULT_OVERCURRENT CURRENT_RAW ");
  PC_SERIAL.println(cur_raw);
}

void clearFault() {
  fault_latched = false;
  current_fault_start_time = 0;
  PC_SERIAL.println("OK_CLEAR_FAULT");
}

bool configureDynamixel() {
  dxl_ready = false;
  current_based_mode = false;
  current_fault_start_time = 0;

  restoreDxlPort();
  delay(100);

  if (!dxl.ping(DXL_ID)) {
    PC_SERIAL.print("ERR_DYNAMIXEL_NOT_FOUND");
    printDxlConfig();
    PC_SERIAL.println();
    return false;
  }

  PC_SERIAL.print("DYNAMIXEL_FOUND");
  printDxlConfig();
  PC_SERIAL.print(" MODEL ");
  PC_SERIAL.println(dxl.getModelNumber(DXL_ID));

  dxl.torqueOff(DXL_ID);

  actual_current_limit_raw = dxl.readControlTableItem(CURRENT_LIMIT, DXL_ID);
  if (actual_current_limit_raw != MAX_GOAL_CURRENT_RAW) {
    if (dxl.writeControlTableItem(CURRENT_LIMIT, DXL_ID, MAX_GOAL_CURRENT_RAW)) {
      actual_current_limit_raw = dxl.readControlTableItem(CURRENT_LIMIT, DXL_ID);
      PC_SERIAL.print("ACTUAL_CURRENT_LIMIT_RAW ");
      PC_SERIAL.println(actual_current_limit_raw);
    } else {
      PC_SERIAL.println("WARN_CURRENT_LIMIT_WRITE_FAILED");
    }
  } else {
    PC_SERIAL.print("ACTUAL_CURRENT_LIMIT_RAW ");
    PC_SERIAL.println(actual_current_limit_raw);
  }

  current_based_mode = dxl.setOperatingMode(DXL_ID, OP_CURRENT_BASED_POSITION);

  if (current_based_mode) {
    PC_SERIAL.println("MODE_CURRENT_BASED_POSITION");
  } else {
    PC_SERIAL.println("WARN_CURRENT_BASED_POSITION_FAILED");

    if (REQUIRE_CURRENT_BASED_POSITION) {
      PC_SERIAL.println("ERR_CURRENT_BASED_POSITION_REQUIRED");
      return false;
    }

    PC_SERIAL.println("TRY_POSITION_MODE");

    if (!dxl.setOperatingMode(DXL_ID, OP_POSITION)) {
      PC_SERIAL.println("ERR_OPERATING_MODE_FAILED");
      return false;
    }

    PC_SERIAL.println("MODE_POSITION");
  }

  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, PROFILE_VELOCITY_RAW);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, PROFILE_ACCELERATION_RAW);

  if (!dxl.torqueOn(DXL_ID)) {
    PC_SERIAL.println("ERR_TORQUE_ON_FAILED");
    return false;
  }

  dxl_ready = true;
  fault_latched = false;
  openGripper();

  return true;
}

bool setupDynamixel() {
  enableDxlPower();
  return configureDynamixel();
}

bool configureLift() {
  if (!dxl.ping(LIFT_DXL_ID)) {
    PC_SERIAL.print("ERR_LIFT_NOT_FOUND ID ");
    PC_SERIAL.println(LIFT_DXL_ID);
    lift_ready = false;
    return false;
  }

  dxl.torqueOff(LIFT_DXL_ID);

  if (!dxl.setOperatingMode(LIFT_DXL_ID, OP_CURRENT_BASED_POSITION)) {
    PC_SERIAL.println("ERR_LIFT_MODE_FAILED");
    lift_ready = false;
    return false;
  }

  dxl.writeControlTableItem(PROFILE_VELOCITY, LIFT_DXL_ID, LIFT_PROFILE_VELOCITY_RAW);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, LIFT_DXL_ID, LIFT_PROFILE_ACCEL_RAW);

  if (!dxl.torqueOn(LIFT_DXL_ID)) {
    PC_SERIAL.println("ERR_LIFT_TORQUE_ON_FAILED");
    lift_ready = false;
    return false;
  }

  dxl.setGoalCurrent(LIFT_DXL_ID, LIFT_GOAL_CURRENT_RAW);
  lift_ready = true;

  // 이 DXL 전원 세션에서 처음 초기화되는 순간의 위치 = 바닥 홈으로 간주(1회).
  // LIFT_TORQUE_OFF 등으로 lift_ready만 내려간 재초기화에서는 홈을 다시 잡지 않는다.
  if (!lift_home_set) {
    lift_home_raw = (int32_t)dxl.getPresentPosition(LIFT_DXL_ID);
    lift_home_set = true;
    PC_SERIAL.print("LIFT_HOME_SET ");
    PC_SERIAL.println(lift_home_raw);
  }

  PC_SERIAL.print("LIFT_READY ID ");
  PC_SERIAL.print(LIFT_DXL_ID);
  PC_SERIAL.print(" MODEL ");
  PC_SERIAL.println(dxl.getModelNumber(LIFT_DXL_ID));
  return true;
}

bool ensureLift() {
  if (lift_ready && dxl.ping(LIFT_DXL_ID)) {
    return true;
  }
  return configureLift();
}

void printLiftStatus() {
  if (!dxl.ping(LIFT_DXL_ID)) {
    PC_SERIAL.println("ERR_LIFT_NOT_FOUND");
    return;
  }
  int32_t pos_raw = (int32_t)dxl.getPresentPosition(LIFT_DXL_ID);
  int32_t cur_raw = (int32_t)dxl.getPresentCurrent(LIFT_DXL_ID);
  int32_t moving = dxl.readControlTableItem(MOVING, LIFT_DXL_ID);
  int32_t err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, LIFT_DXL_ID);
  PC_SERIAL.print("LIFT_STATUS READY ");
  PC_SERIAL.print(lift_ready ? 1 : 0);
  PC_SERIAL.print(" POS_RAW ");
  PC_SERIAL.print(pos_raw);
  PC_SERIAL.print(" CURRENT_RAW ");
  PC_SERIAL.print(cur_raw);
  PC_SERIAL.print(" MOVING ");
  PC_SERIAL.print(moving);
  PC_SERIAL.print(" HW_ERROR ");
  PC_SERIAL.print(err);
  PC_SERIAL.print(" TORQUE ");
  PC_SERIAL.print(dxl.getTorqueEnableStat(LIFT_DXL_ID) ? 1 : 0);
  PC_SERIAL.print(" HOME_SET ");
  PC_SERIAL.print(lift_home_set ? 1 : 0);
  PC_SERIAL.print(" HOME_RAW ");
  PC_SERIAL.print(lift_home_raw);
  PC_SERIAL.print(" STROKE ");
  PC_SERIAL.println(LIFT_STROKE_TICKS);
}

// 홈(바닥) 기준 절대 목표로 이동. delta_from_home = 홈에서의 상대 tick.
void liftGoFromHome(long delta_from_home, const char* tag) {
  if (!ensureLift()) {
    return;
  }
  int32_t present = (int32_t)dxl.getPresentPosition(LIFT_DXL_ID);
  int32_t goal = lift_home_raw + (int32_t)delta_from_home;
  long travel = (long)goal - (long)present;
  if (travel > LIFT_MAX_MOVE_TICKS || travel < -LIFT_MAX_MOVE_TICKS) {
    PC_SERIAL.print("ERR_LIFT_MOVE_TOO_FAR ");
    PC_SERIAL.println(travel);
    return;
  }
  dxl.setGoalCurrent(LIFT_DXL_ID, LIFT_GOAL_CURRENT_RAW);
  bool ok = dxl.setGoalPosition(LIFT_DXL_ID, (float)goal);
  PC_SERIAL.print(ok ? "OK_" : "ERR_");
  PC_SERIAL.print(tag);
  PC_SERIAL.print(" HOME ");
  PC_SERIAL.print(lift_home_raw);
  PC_SERIAL.print(" ");
  PC_SERIAL.print(present);
  PC_SERIAL.print(" -> ");
  PC_SERIAL.println(goal);
}

// 최고점: 홈에서 (스트로크 - 마진)만큼 올림. 하드 스톱에 박지 않도록 여유를 둔다.
void liftToTop() {
  liftGoFromHome(LIFT_RAISE_SIGN * (LIFT_STROKE_TICKS - LIFT_TOP_MARGIN_TICKS),
                 "LIFT_TO_TOP");
}

// 중간점: 최고점 목표의 절반 (홈 기준 절대라 mid<->top<->bottom 어떤 순서든
// 안전 — 매 명령이 present 를 읽어 절대 goal 로 이동, 상대 스트로크 아님).
void liftToMid() {
  liftGoFromHome(LIFT_RAISE_SIGN * (LIFT_STROKE_TICKS - LIFT_TOP_MARGIN_TICKS) / 2,
                 "LIFT_TO_MID");
}

// 바닥: 홈 복귀.
void liftToBottom() {
  liftGoFromHome(0, "LIFT_TO_BOTTOM");
}

// 현재 위치를 바닥 홈으로 재지정(부팅이 바닥이 아니었을 때 수동 보정).
void liftSetHome() {
  if (!ensureLift()) {
    return;
  }
  lift_home_raw = (int32_t)dxl.getPresentPosition(LIFT_DXL_ID);
  lift_home_set = true;
  PC_SERIAL.print("OK_LIFT_SET_HOME ");
  PC_SERIAL.println(lift_home_raw);
}

void liftMove(long delta_ticks) {
  if (delta_ticks > LIFT_MAX_MOVE_TICKS || delta_ticks < -LIFT_MAX_MOVE_TICKS) {
    PC_SERIAL.println("ERR_LIFT_MOVE_TOO_FAR");
    return;
  }
  if (!ensureLift()) {
    return;
  }
  int32_t present = (int32_t)dxl.getPresentPosition(LIFT_DXL_ID);
  int32_t goal = present + (int32_t)delta_ticks;
  dxl.setGoalCurrent(LIFT_DXL_ID, LIFT_GOAL_CURRENT_RAW);
  bool ok = dxl.setGoalPosition(LIFT_DXL_ID, (float)goal);
  PC_SERIAL.print(ok ? "OK_LIFT_MOVE " : "ERR_LIFT_MOVE ");
  PC_SERIAL.print(present);
  PC_SERIAL.print(" -> ");
  PC_SERIAL.println(goal);
}

void liftStop() {
  if (!dxl.ping(LIFT_DXL_ID)) {
    PC_SERIAL.println("ERR_LIFT_NOT_FOUND");
    return;
  }
  int32_t present = (int32_t)dxl.getPresentPosition(LIFT_DXL_ID);
  dxl.setGoalPosition(LIFT_DXL_ID, (float)present);
  PC_SERIAL.print("OK_LIFT_STOP ");
  PC_SERIAL.println(present);
}

void liftTorqueOff() {
  // 손 조작/캘리브레이션용. 전원 유지 상태라 멀티턴 위치 추적은 계속된다.
  if (!dxl.ping(LIFT_DXL_ID)) {
    PC_SERIAL.println("ERR_LIFT_NOT_FOUND");
    return;
  }
  dxl.torqueOff(LIFT_DXL_ID);
  lift_ready = false;  // 다음 LIFT_MOVE 때 configureLift로 토크 재인가
  PC_SERIAL.print("OK_LIFT_TORQUE_OFF POS_RAW ");
  PC_SERIAL.println((int32_t)dxl.getPresentPosition(LIFT_DXL_ID));
}

void liftTorqueOn() {
  if (!ensureLift()) {
    return;
  }
  PC_SERIAL.print("OK_LIFT_TORQUE_ON POS_RAW ");
  PC_SERIAL.println((int32_t)dxl.getPresentPosition(LIFT_DXL_ID));
}

void printScanFound(uint8_t id, float protocol, uint32_t baud) {
  PC_SERIAL.print("SCAN_FOUND ID ");
  PC_SERIAL.print(id);
  PC_SERIAL.print(" MODEL ");
  PC_SERIAL.print(dxl.getModelNumber(id));
  PC_SERIAL.print(" PROTOCOL ");
  PC_SERIAL.print(protocol);
  PC_SERIAL.print(" BAUD ");
  PC_SERIAL.println(baud);
}

void scanDynamixel(uint8_t max_id) {
  int found_count = 0;

  PC_SERIAL.print("SCAN_BEGIN MAX_ID ");
  PC_SERIAL.println(max_id);

  for (int protocol = 1; protocol <= 2; protocol++) {
    dxl.setPortProtocolVersion((float)protocol);

    for (size_t baud_index = 0; baud_index < sizeof(SCAN_BAUDS) / sizeof(SCAN_BAUDS[0]); baud_index++) {
      uint32_t baud = SCAN_BAUDS[baud_index];
      dxl.begin(baud);

      PC_SERIAL.print("SCAN_TRY PROTOCOL ");
      PC_SERIAL.print(protocol);
      PC_SERIAL.print(" BAUD ");
      PC_SERIAL.println(baud);

      for (uint8_t id = 0; id <= max_id; id++) {
        if (dxl.ping(id)) {
          printScanFound(id, (float)protocol, baud);
          found_count++;
        }
      }
    }
  }

  restoreDxlPort();

  if (found_count == 0) {
    PC_SERIAL.println("SCAN_NONE");
  }

  PC_SERIAL.print("SCAN_DONE COUNT ");
  PC_SERIAL.println(found_count);
}

bool parseSetDxl(String cmd, uint8_t *id, uint32_t *baud, float *protocol) {
  String rest = cmd.substring(8);
  rest.trim();

  if (rest.length() == 0) {
    return false;
  }

  int first_space = rest.indexOf(' ');
  String id_text = first_space < 0 ? rest : rest.substring(0, first_space);

  if (!isUnsignedIntegerText(id_text)) {
    return false;
  }

  long parsed_id = id_text.toInt();

  if (parsed_id < 0 || parsed_id >= DXL_BROADCAST_ID) {
    return false;
  }

  *id = (uint8_t)parsed_id;
  *baud = DXL_BAUD;
  *protocol = DXL_PROTOCOL_VERSION;

  if (first_space < 0) {
    return true;
  }

  rest = rest.substring(first_space + 1);
  rest.trim();
  if (rest.length() == 0) {
    return true;
  }

  int second_space = rest.indexOf(' ');
  String baud_text = second_space < 0 ? rest : rest.substring(0, second_space);

  if (!isUnsignedIntegerText(baud_text)) {
    return false;
  }

  long parsed_baud = baud_text.toInt();

  if (parsed_baud <= 0) {
    return false;
  }

  *baud = (uint32_t)parsed_baud;

  if (second_space < 0) {
    return true;
  }

  rest = rest.substring(second_space + 1);
  rest.trim();
  if (rest.length() == 0) {
    return true;
  }

  float parsed_protocol = rest.toFloat();

  if (parsed_protocol != 1.0 && parsed_protocol != 2.0) {
    return false;
  }

  *protocol = parsed_protocol;
  return true;
}

void setDxlConfig(String cmd) {
  uint8_t id = DXL_ID;
  uint32_t baud = DXL_BAUD;
  float protocol = DXL_PROTOCOL_VERSION;

  if (!parseSetDxl(cmd, &id, &baud, &protocol)) {
    PC_SERIAL.println("ERR_BAD_SET_DXL_COMMAND");
    return;
  }

  DXL_ID = id;
  DXL_BAUD = baud;
  DXL_PROTOCOL_VERSION = protocol;

  bool ok = configureDynamixel();

  if (ok) {
    PC_SERIAL.print("OK_SET_DXL");
  } else {
    PC_SERIAL.print("ERR_SET_DXL");
  }

  printDxlConfig();
  PC_SERIAL.println();
}

void setup() {
  PC_SERIAL.begin(115200);
  PC_SERIAL.setTimeout(30);

  unsigned long start_time = millis();

  while (!PC_SERIAL && millis() - start_time < 3000) {
  }

  PC_SERIAL.println("OPENRB_GRIPPER_CONTROLLER_BOOT");
  PC_SERIAL.print("SAFE_RANGE_DEG ");
  PC_SERIAL.print(CLOSED_DEG);
  PC_SERIAL.print(" TO ");
  PC_SERIAL.println(FULL_OPEN_DEG);
  PC_SERIAL.print("SAFE_ARC_DEG ");
  PC_SERIAL.print(CLOSED_DEG);
  PC_SERIAL.print(" TO ");
  PC_SERIAL.print(FULL_OPEN_DEG);
  PC_SERIAL.print(" WRAP ");
  PC_SERIAL.println(safeArcWrapsZero() ? 1 : 0);
  PC_SERIAL.print("CURRENT_LIMIT_RAW ");
  PC_SERIAL.println(MAX_GOAL_CURRENT_RAW);
  PC_SERIAL.print("DXL_POWER_ON_BOOT ");
  PC_SERIAL.println(ENABLE_DXL_POWER_ON_BOOT ? 1 : 0);

  bool ok = setupDynamixel();

  if (ok) {
    PC_SERIAL.println("READY");
  } else {
    PC_SERIAL.println("NOT_READY");
  }

  last_cmd_time = millis();
}

void loop() {
  if (PC_SERIAL.available()) {
    String cmd = PC_SERIAL.readStringUntil('\n');
    cmd.trim();

    last_cmd_time = millis();

    if (cmd == "OPEN") {
      openGripper();
    }

    else if (cmd == "CLOSE") {
      closeGripper();
    }

    else if (cmd.startsWith("SET_DEG ")) {
      float deg = cmd.substring(8).toFloat();
      moveToDeg(deg);
    }

    else if (cmd.startsWith("SET_RATIO ")) {
      float ratio = cmd.substring(10).toFloat();
      moveToRatio(ratio);
    }

    else if (cmd.startsWith("GRIP ")) {
      int first_space = cmd.indexOf(' ');
      int second_space = cmd.indexOf(' ', first_space + 1);

      if (second_space > 0) {
        float deg = cmd.substring(first_space + 1, second_space).toFloat();
        int cur = cmd.substring(second_space + 1).toInt();

        cur = clampCurrentRaw(cur);
        GRIP_CURRENT_RAW = cur;

        moveToDeg(deg, GRIP_CURRENT_RAW);
      } else {
        PC_SERIAL.println("ERR_BAD_GRIP_COMMAND");
      }
    }

    else if (cmd.startsWith("SET_CURRENT ")) {
      GRIP_CURRENT_RAW = clampCurrentRaw(cmd.substring(12).toInt());
      PC_SERIAL.print("OK_SET_CURRENT CURRENT_RAW ");
      PC_SERIAL.println(GRIP_CURRENT_RAW);
    }

    else if (cmd == "STATUS?") {
      printStatus();
    }

    else if (cmd == "STOP") {
      if (dxl_ready && dxl.torqueOff(DXL_ID)) {
        PC_SERIAL.println("OK_STOP");
      } else {
        PC_SERIAL.println("ERR_STOP");
      }
    }

    else if (cmd == "TORQUE_ON") {
      if (fault_latched) {
        PC_SERIAL.println("ERR_FAULT_LATCHED");
      } else if (dxl_ready && dxl.torqueOn(DXL_ID)) {
        PC_SERIAL.println("OK_TORQUE_ON");
      } else {
        PC_SERIAL.println("ERR_TORQUE_ON");
      }
    }

    else if (cmd == "REBOOT") {
      fault_latched = false;
      if (dxl_ready) {
        dxl.reboot(DXL_ID);
        delay(1000);
      }

      bool ok = setupDynamixel();

      if (ok) {
        PC_SERIAL.println("OK_REBOOT");
      } else {
        PC_SERIAL.println("ERR_REBOOT");
      }
    }

    else if (cmd == "DXL_POWER_ON") {
      setDxlPower(true);
      bool ok = configureDynamixel();
      PC_SERIAL.println(ok ? "OK_DXL_POWER_ON" : "ERR_DXL_POWER_ON");
    }

    else if (cmd == "DXL_POWER_OFF") {
      if (dxl_ready) {
        dxl.torqueOff(DXL_ID);
      }
      dxl_ready = false;
      fault_latched = false;
      lift_ready = false;
      lift_home_set = false;  // 전원 끊기면 멀티턴 리셋 → 홈 재캡처 필요
      setDxlPower(false);
      PC_SERIAL.println("OK_DXL_POWER_OFF");
    }

    else if (cmd == "DXL_POWER_CYCLE") {
      if (dxl_ready) {
        dxl.torqueOff(DXL_ID);
      }
      dxl_ready = false;
      fault_latched = false;
      lift_ready = false;
      lift_home_set = false;  // 전원 끊기면 멀티턴 리셋 → 홈 재캡처 필요
      setDxlPower(false);
      delay(500);
      setDxlPower(true);
      bool ok = configureDynamixel();
      PC_SERIAL.println(ok ? "OK_DXL_POWER_CYCLE" : "ERR_DXL_POWER_CYCLE");
    }

    else if (cmd == "CLEAR_FAULT") {
      clearFault();
    }

    else if (cmd == "PING") {
      if (dxl.ping(DXL_ID)) {
        PC_SERIAL.print("OK_PING");
        printDxlConfig();
        PC_SERIAL.print(" MODEL ");
        PC_SERIAL.println(dxl.getModelNumber(DXL_ID));

        if (!dxl_ready) {
          configureDynamixel();
        }
      } else {
        dxl_ready = false;
        PC_SERIAL.print("ERR_PING");
        printDxlConfig();
        PC_SERIAL.println();
      }
    }

    else if (cmd == "SCAN") {
      scanDynamixel(DEFAULT_SCAN_MAX_ID);
    }

    else if (cmd.startsWith("SCAN ")) {
      String max_id_text = cmd.substring(5);
      max_id_text.trim();

      if (!isUnsignedIntegerText(max_id_text)) {
        PC_SERIAL.println("ERR_BAD_SCAN_COMMAND");
      } else if (max_id_text.toInt() >= DXL_BROADCAST_ID) {
        PC_SERIAL.println("ERR_BAD_SCAN_COMMAND");
      } else {
        scanDynamixel((uint8_t)max_id_text.toInt());
      }
    }

    else if (cmd.startsWith("SET_DXL ")) {
      setDxlConfig(cmd);
    }

    else if (cmd.startsWith("LIFT_MOVE ")) {
      liftMove(cmd.substring(10).toInt());
    }

    else if (cmd == "LIFT_STATUS?") {
      printLiftStatus();
    }

    else if (cmd == "LIFT_STOP") {
      liftStop();
    }

    else if (cmd == "LIFT_TORQUE_OFF") {
      liftTorqueOff();
    }

    else if (cmd == "LIFT_TORQUE_ON") {
      liftTorqueOn();
    }

    else if (cmd == "LIFT_TO_TOP") {
      liftToTop();
    }

    else if (cmd == "LIFT_TO_MID") {
      liftToMid();
    }

    else if (cmd == "LIFT_TO_BOTTOM") {
      liftToBottom();
    }

    else if (cmd == "LIFT_SET_HOME") {
      liftSetHome();
    }

    else {
      PC_SERIAL.println("ERR_UNKNOWN_COMMAND");
    }

    last_cmd_time = millis();
  }

  checkCurrentFault();

  if (millis() - last_cmd_time > WATCHDOG_MS) {
    if (!fault_latched) {
      openGripper();
    }
    last_cmd_time = millis();
    PC_SERIAL.println("WATCHDOG_OPEN");
  }
}
