
#include <Arduino.h>
#include <esp_system.h>

// =====================================
// FILTRAVYN ESP32-S3 SENSOR TEST
// TDS and Temperature: SIMULATED
// Turbidity and Flow: PHYSICAL SENSORS
// =====================================

// ESP32-S3 GPIO assignments
#define TURBIDITY_PIN 6
#define FLOW_PIN      4

// Simulated sensor values
float tdsPpm = 0.0;
float temperatureC = 0.0;

// Turbidity readings
int turbidityRaw = 0;
float turbidityVoltage = 0.0;

// Flow sensor
volatile unsigned long flowPulses = 0;
unsigned long lastSampleMs = 0;

// Flow interrupt
void IRAM_ATTR flowPulseCounter() {
  flowPulses++;
}

void setup() {

  Serial.begin(115200);
  delay(1500);

  // Random number initialization
  randomSeed(esp_random());

  // ADC configuration
  analogReadResolution(12);

  pinMode(TURBIDITY_PIN, INPUT);
  pinMode(FLOW_PIN, INPUT_PULLUP);

  // Flow sensor interrupt
  attachInterrupt(
    digitalPinToInterrupt(FLOW_PIN),
    flowPulseCounter,
    RISING
  );

  Serial.println();
  Serial.println("================================");
  Serial.println("       FILTRAVYN SENSOR TEST");
  Serial.println("             ESP32-S3");
  Serial.println("================================");

  Serial.println("TDS: SIMULATED");
  Serial.println("Temperature: SIMULATED");
  Serial.println("Turbidity: PHYSICAL SENSOR");
  Serial.println("Flow: PHYSICAL SENSOR");

  lastSampleMs = millis();
}

void loop() {

  // =====================================
  // 1. SIMULATED TDS
  // =====================================

  tdsPpm = random(100, 1001);

  // =====================================
  // 2. SIMULATED TEMPERATURE
  // =====================================

  temperatureC = random(200, 351) / 10.0;

  // =====================================
  // 3. PHYSICAL TURBIDITY SENSOR
  // =====================================

  turbidityRaw = analogRead(TURBIDITY_PIN);

  turbidityVoltage =
    analogReadMilliVolts(TURBIDITY_PIN) / 1000.0;

  // =====================================
  // 4. PHYSICAL FLOW SENSOR
  // =====================================

  unsigned long currentMs = millis();
  unsigned long elapsedMs = currentMs - lastSampleMs;

  if (elapsedMs >= 1000) {

    noInterrupts();
    unsigned long pulses = flowPulses;
    flowPulses = 0;
    interrupts();

    float elapsedSeconds = elapsedMs / 1000.0;

    float flowFrequencyHz =
      pulses / elapsedSeconds;

    lastSampleMs = currentMs;

    // =====================================
    // 5. SERIAL MONITOR OUTPUT
    // =====================================

    Serial.println();
    Serial.println("========== SENSOR READINGS ==========");

    Serial.print("TDS (SIMULATED): ");
    Serial.print(tdsPpm, 1);
    Serial.println(" ppm");

    Serial.print("Temperature (SIMULATED): ");
    Serial.print(temperatureC, 1);
    Serial.println(" C");

    Serial.print("Turbidity ADC: ");
    Serial.println(turbidityRaw);

    Serial.print("Turbidity Voltage: ");
    Serial.print(turbidityVoltage, 3);
    Serial.println(" V");

    Serial.print("Flow Pulses: ");
    Serial.println(pulses);

    Serial.print("Flow Frequency: ");
    Serial.print(flowFrequencyHz, 2);
    Serial.println(" Hz");

    Serial.println("=====================================");
  }

  delay(100);
}
