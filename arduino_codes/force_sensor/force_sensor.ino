#include "HX711.h"

#define DOUT  6
#define CLK  5

HX711 scale;

float calibration_factor = 208;  // Update this after calibration (may be negative)
float current_reading;

void setup() {
  Serial.begin(115200);
  scale.begin(DOUT, CLK);
  scale.set_scale(calibration_factor);  // Set scale here once
  scale.tare();  // Zero the scale with no load
  Serial.println("Load cell ready. Readings in Newtons (N):");
}

float runningAverage(float M) {
  #define LM_SIZE 4  // Increased for better smoothing (adjust as needed)
  static float LM[LM_SIZE];
  static byte index = 0;
  static float sum = 0;
  static byte count = 0;

  sum -= LM[index];
  LM[index] = M;
  sum += LM[index];
  index = (index + 1) % LM_SIZE;
  if (count < LM_SIZE) count++;

  return sum / count;
}

void loop() {
  current_reading = scale.get_units();  // Gets scaled reading (in grams if calibrated that way)
  float smoothed_reading = runningAverage(current_reading);
  float force_N = (smoothed_reading * 9.81) / 1000.0;  // Convert grams to kg, then to N (compressive force)

  Serial.println(force_N, 4);
//   Serial.println(smoothed_reading, 4);
  
  // Serial.print("Force: ");
  // Serial.print(force_N, 4);  // Print with 4 decimal places
  // Serial.println(" N");
  
  delay(1);  // Slower update for readability (adjust as needed)
}
