/*
 * hand_controller.ino — Pilotage d'une main robotique à 5 servos
 * ----------------------------------------------------------------
 * Reçoit depuis le PC (main.py --mode serial), à 115200 bauds, des trames :
 *
 *     $a0,a1,a2,a3,a4\n          doigts seuls (wrist.enabled: false)
 *     $a0,a1,a2,a3,a4,a5,a6\n    doigts + poignet (wrist.enabled: true)
 *
 * Ordre : pouce, index, majeur, annulaire, auriculaire, rotation du poignet,
 * inclinaison du poignet. Angles entiers 0–180. Avec une trame à 5 valeurs,
 * les servos du poignet restent en position neutre.
 *
 * Sécurités :
 *  - trame validée (début '$', 5 ou 7 entiers 0–180, longueur bornée) ;
 *  - limitation de vitesse : chaque servo se déplace d'au plus
 *    MAX_SPEED_DEG_PER_S degrés par seconde (protège la mécanique et les fils) ;
 *  - timeout : sans trame valide pendant TIMEOUT_MS, la main revient
 *    doucement en position « main ouverte ».
 *
 * Cartes testées : Arduino Uno / Nano (bibliothèque Servo.h intégrée).
 * ESP32 : remplacer <Servo.h> par <ESP32Servo.h> (bibliothèque « ESP32Servo »).
 *
 * IMPORTANT : alimenter les servos avec une alimentation 5–6 V SÉPARÉE
 * (≥ 1 A par MG90S, ≥ 2,5 A par MG996R), et relier sa masse (GND) à celle
 * de l'Arduino. Ne jamais alimenter 5 servos depuis la broche 5V de l'Arduino.
 */

#include <Servo.h>

// ----------------------------------------------------------- Réglages
const uint8_t NUM_FINGERS = 5;
const uint8_t NUM_SERVOS = 7;   // 5 doigts + rotation + inclinaison du poignet

// Broches des servos : pouce, index, majeur, annulaire, auriculaire,
// rotation du poignet, inclinaison du poignet. (Servo.h accepte n'importe
// quelle broche numérique.) Sans poignet, laissez simplement D11/D12 libres.
const uint8_t SERVO_PINS[NUM_SERVOS] = {3, 5, 6, 9, 10, 11, 12};

// Position de repos de chaque servo, utilisée au démarrage et en cas de timeout.
//  - doigts : angle « main ouverte » = 0 % dans config.yaml, c'est-à-dire
//    min_angles (ou max_angles si le doigt est inversé) ;
//  - poignet : position neutre = milieu de [min_angle, max_angle] de
//    wrist.rotation / wrist.deviation (90 avec les réglages par défaut).
const uint8_t OPEN_ANGLES[NUM_SERVOS] = {0, 0, 0, 0, 0, 90, 90};

// Bornes de sécurité mécanique par servo (en plus de celles du PC).
// Inclinaison : la course d'un vrai poignet est faible, on la borne à 45–135°.
const uint8_t SAFE_MIN[NUM_SERVOS] = {0, 0, 0, 0, 0, 0, 45};
const uint8_t SAFE_MAX[NUM_SERVOS] = {180, 180, 180, 180, 180, 180, 135};

const unsigned long BAUD_RATE = 115200;
const unsigned long TIMEOUT_MS = 1000;        // Main ouverte si plus de trame pendant 1 s
const float MAX_SPEED_DEG_PER_S = 400.0;      // Vitesse max d'un servo
const unsigned long UPDATE_PERIOD_MS = 10;    // Période de mise à jour des servos (100 Hz)
const uint8_t LINE_MAX = 40;                  // "$180,180,180,180,180,180,180" = 28 caractères

// ------------------------------------------------------------- État
Servo servos[NUM_SERVOS];
float currentAngle[NUM_SERVOS];   // Angle réellement commandé (après limitation de vitesse)
uint8_t targetAngle[NUM_SERVOS];  // Dernier angle demandé par le PC

char lineBuf[LINE_MAX];
uint8_t lineLen = 0;
bool lineOverflow = false;

unsigned long lastFrameMs = 0;
unsigned long lastUpdateMs = 0;
bool timedOut = true;

// ------------------------------------------------- Analyse de trame
// Analyse "$a0,a1,...,aN" (sans le '\n'). Renvoie le nombre de valeurs lues
// (NUM_FINGERS ou NUM_SERVOS) et remplit `out`, ou 0 si la trame est invalide.
uint8_t parseFrame(const char *s, uint8_t out[NUM_SERVOS]) {
  if (*s != '$') return 0;
  s++;
  uint8_t count = 0;
  while (true) {
    if (count >= NUM_SERVOS) return 0;        // trop de valeurs
    if (*s < '0' || *s > '9') return 0;       // au moins un chiffre
    int value = 0;
    uint8_t digits = 0;
    while (*s >= '0' && *s <= '9') {
      value = value * 10 + (*s - '0');
      if (++digits > 3) return 0;
      s++;
    }
    if (value > 180) return 0;
    out[count++] = (uint8_t)value;
    if (*s == ',') {                          // une autre valeur suit
      s++;
      continue;
    }
    if (*s == '\r') s++;                      // tolère les fins de ligne Windows
    if (*s != '\0') return 0;                 // caractère inattendu
    break;
  }
  return (count == NUM_FINGERS || count == NUM_SERVOS) ? count : 0;
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '$') {             // début de trame : on (re)commence
      lineLen = 0;
      lineOverflow = false;
    }
    if (c == '\n') {
      if (!lineOverflow && lineLen > 0) {
        lineBuf[lineLen] = '\0';
        uint8_t angles[NUM_SERVOS];
        uint8_t count = parseFrame(lineBuf, angles);
        if (count > 0) {
          // Trame à 5 valeurs : les servos du poignet gardent leur consigne.
          for (uint8_t i = 0; i < count; i++) {
            targetAngle[i] = constrain(angles[i], SAFE_MIN[i], SAFE_MAX[i]);
          }
          lastFrameMs = millis();
          timedOut = false;
        }
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < LINE_MAX - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineOverflow = true;      // ligne trop longue : ignorée jusqu'au prochain '\n'
    }
  }
}

// ------------------------------------------------ Mise à jour servos
void updateServos() {
  const float maxStep = MAX_SPEED_DEG_PER_S * UPDATE_PERIOD_MS / 1000.0;
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    float delta = (float)targetAngle[i] - currentAngle[i];
    if (delta > maxStep) delta = maxStep;
    if (delta < -maxStep) delta = -maxStep;
    currentAngle[i] += delta;
    servos[i].write((int)(currentAngle[i] + 0.5));
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    currentAngle[i] = OPEN_ANGLES[i];
    targetAngle[i] = OPEN_ANGLES[i];
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(OPEN_ANGLES[i]);
  }
  lastUpdateMs = millis();
}

void loop() {
  readSerial();

  unsigned long now = millis();
  // Timeout de sécurité : plus de trame → main ouverte.
  if (!timedOut && now - lastFrameMs > TIMEOUT_MS) {
    timedOut = true;
    for (uint8_t i = 0; i < NUM_SERVOS; i++) targetAngle[i] = OPEN_ANGLES[i];
  }
  if (now - lastUpdateMs >= UPDATE_PERIOD_MS) {
    lastUpdateMs = now;
    updateServos();
  }
}

/* =====================================================================
 * VARIANTE AVEC CARTE PCA9685 (16 voies PWM en I2C)
 * ---------------------------------------------------------------------
 * Avantages : libère les broches de l'Arduino, PWM plus stable, jusqu'à
 * 16 servos. Câblage : SDA → A4, SCL → A5 (Uno/Nano), VCC → 5V Arduino
 * (logique), V+ (bornier) → alimentation 5–6 V des servos, GND communs.
 * Bibliothèque : « Adafruit PWM Servo Driver Library ».
 *
 * Remplacer :
 *
 *   #include <Servo.h>
 *   Servo servos[NUM_SERVOS];
 *
 * par :
 *
 *   #include <Wire.h>
 *   #include <Adafruit_PWMServoDriver.h>
 *   Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);
 *   const uint8_t PCA_CHANNELS[NUM_SERVOS] = {0, 1, 2, 3, 4, 5, 6};
 *   // Largeurs d'impulsion (µs) pour 0° et 180° : à ajuster selon vos servos.
 *   const uint16_t PULSE_MIN_US = 500, PULSE_MAX_US = 2400;
 *
 *   void writeServo(uint8_t i, float angle) {
 *     uint16_t us = PULSE_MIN_US + (uint16_t)((PULSE_MAX_US - PULSE_MIN_US) * angle / 180.0);
 *     pca.writeMicroseconds(PCA_CHANNELS[i], us);
 *   }
 *
 * Dans setup(), remplacer la boucle attach()/write() par :
 *
 *   pca.begin();
 *   pca.setOscillatorFrequency(27000000);
 *   pca.setPWMFreq(50);               // servos analogiques : 50 Hz
 *   for (uint8_t i = 0; i < NUM_SERVOS; i++) {
 *     currentAngle[i] = targetAngle[i] = OPEN_ANGLES[i];
 *     writeServo(i, OPEN_ANGLES[i]);
 *   }
 *
 * Dans updateServos(), remplacer servos[i].write(...) par :
 *
 *   writeServo(i, currentAngle[i]);
 * ===================================================================== */
