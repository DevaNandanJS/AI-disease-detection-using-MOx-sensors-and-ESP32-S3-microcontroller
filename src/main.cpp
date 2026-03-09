#include <Arduino.h>
#include <LittleFS.h>
#include "esp_system.h" // For heap size checking
#include "esp_timer.h"  // For high-resolution timing
#include "xgb_model.h"  // Our transpiled XGBoost model

// --- CRITICAL: USER CONFIGURATION ---
// You MUST set this to the number of features your model expects.
// This value must match the number of columns in your test.csv.
const int NUM_FEATURES = 10; // <--- UPDATE THIS VALUE

const char* TEST_DATASET_PATH = "/test.csv";

// --- Function Prototypes ---
bool parse_csv_line(String& line, float* features);

void setup() {
    Serial.begin(115200);
    // Wait up to 2 seconds for the serial port to connect
    for (int i = 0; i < 20 && !Serial; ++i) {
        delay(100);
    }

    Serial.println("
--- ESP32-S3 XGBoost Inference Benchmark ---");

    // 1. Mount the LittleFS filesystem
    if (!LittleFS.begin()) {
        Serial.println("FATAL: Failed to mount LittleFS. Check partitioning and ensure data directory was flashed.");
        while (1) yield(); // Halt
    }
    Serial.println("LittleFS mounted successfully.");

    // 2. Open the test dataset
    File dataFile = LittleFS.open(TEST_DATASET_PATH);
    if (!dataFile) {
        Serial.printf("FATAL: Failed to open %s. Did you place test.csv in the 'data' directory?
", TEST_DATASET_PATH);
        while (1) yield(); // Halt
    }
    Serial.printf("Successfully opened %s
", TEST_DATASET_PATH);

    float features[NUM_FEATURES];
    int line_count = 0;
    long total_latency = 0;

    // 3. Process the dataset line by line
    while (dataFile.available()) {
        String line = dataFile.readStringUntil('
');
        line.trim();
        if (line.length() == 0) continue;

        line_count++;

        // 4. Parse the CSV line into a float array
        if (!parse_csv_line(line, features)) {
            Serial.printf("ERROR on line %d: Failed to parse feature vector. Expected %d features. Skipping.
", line_count, NUM_FEATURES);
            continue;
        }

        // --- BENCHMARKING BLOCK ---
        // 5. Measure pre-inference SRAM
        size_t initial_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);

        // 6. Execute inference and measure latency (strictly around the call)
        int64_t start_time = esp_timer_get_time();
        float prediction = score(features); // Call the transpiled model
        int64_t end_time = esp_timer_get_time();

        // 7. Measure post-inference SRAM
        size_t final_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        // --- END BENCHMARKING BLOCK ---

        // 8. Report results for the line
        long latency_us = (long)(end_time - start_time);
        total_latency += latency_us;
        size_t sram_used = (initial_heap > final_heap) ? (initial_heap - final_heap) : 0; // Prevent underflow if memory is allocated

        Serial.printf("L%d: Pred=%.4f, Latency=%ld us, SRAM Used=%u B
", line_count, prediction, latency_us, sram_used);
    }

    dataFile.close();
    Serial.println("
--- Benchmark Summary ---");
    if (line_count > 0) {
        Serial.printf("Total lines processed: %d
", line_count);
        Serial.printf("Average inference latency: %ld us
", total_latency / line_count);
    } else {
        Serial.println("No data lines were processed.");
    }
    Serial.println("--- Benchmark Complete ---");
}

void loop() {
    // Intentionally empty. The benchmark runs once in setup().
    delay(10000);
}

/**
 * @brief Parses a comma-separated string into a float array without dynamic memory allocation.
 *
 * @param line The string line to parse.
 * @param features Pointer to the float array to fill.
 * @return true if parsing was successful (found NUM_FEATURES), false otherwise.
 */
bool parse_csv_line(String& line, float* features) {
    int feature_index = 0;
    int current_pos = 0;
    int next_comma = -1;

    while(feature_index < NUM_FEATURES) {
        next_comma = line.indexOf(',', current_pos);

        if (next_comma == -1) { // Last value
            // Ensure we are at the last feature index
            if (feature_index != NUM_FEATURES - 1) return false;
            features[feature_index++] = line.substring(current_pos).toFloat();
            break;
        }

        features[feature_index++] = line.substring(current_pos, next_comma).toFloat();
        current_pos = next_comma + 1;
    }

    // After the loop, verify we have the exact number of features
    return (feature_index == NUM_FEATURES) && (line.indexOf(',', current_pos) == -1);
}
