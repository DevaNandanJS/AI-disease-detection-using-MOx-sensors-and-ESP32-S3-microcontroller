#include <Arduino.h>
#include <FS.h>
#include <LittleFS.h>
#include <ArduinoJson.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "xgb_model.h"
#include "Benchmark.h"

// --- Configuration ---
const int NUM_FEATURES = 6;
const char* TEST_DATASET_PATH = "/test.csv";

// --- Globals ---
Benchmark benchmark;
StaticJsonDocument<256> json_doc;

// --- Function Prototypes ---
void run_full_benchmark();
bool parse_csv_line(const String& line, float* features, int num_features);

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(10);
    }

    Serial.println("--- ESP32-S3 XGBoost Inference Benchmark ---");
    Serial.println("Send any character to start the benchmark.");

    if (!LittleFS.begin()) {
        Serial.println("FATAL: LittleFS mount failed.");
        while (1) delay(1000);
    }
}

void loop() {
    if (Serial.available() > 0) {
        while(Serial.available()) {
            Serial.read();
        }
        run_full_benchmark();
        Serial.println("\n--- Benchmark Complete ---");
        Serial.println("Send any character to run again.");
    }
}

/**
 * @brief Main benchmark execution function.
 * Integrates the data pipeline, the transpiled model, and the telemetry engine.
 */
void run_full_benchmark() {
    File dataFile = LittleFS.open(TEST_DATASET_PATH);
    if (!dataFile) {
        Serial.printf("ERROR: Failed to open %s.\n", TEST_DATASET_PATH);
        return;
    }

    Serial.printf("Processing %s...\n", TEST_DATASET_PATH);

    static float features[NUM_FEATURES];
    int total_inferences = 0;
    long long total_latency_us = 0;
    size_t peak_internal_sram_usage = 0;
    size_t peak_psram_usage = 0;

    while (dataFile.available()) {
        String line = dataFile.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) continue;

        if (parse_csv_line(line, features, NUM_FEATURES)) {
            total_inferences++;

            benchmark.start();
            score(features); // Call the transpiled XGBoost model
            benchmark.stop();

            const auto& result = benchmark.getResult();
            total_latency_us += result.inference_latency_us;

            size_t internal_ram_consumed = (result.initial_internal_heap > result.final_internal_heap)
                                         ? (result.initial_internal_heap - result.final_internal_heap) : 0;
            size_t psram_consumed = (result.initial_psram > result.final_psram)
                                  ? (result.initial_psram - result.final_psram) : 0;

            if (internal_ram_consumed > peak_internal_sram_usage) {
                peak_internal_sram_usage = internal_ram_consumed;
            }
            if (psram_consumed > peak_psram_usage) {
                peak_psram_usage = psram_consumed;
            }
            
            // Optional: Print per-line JSON report
            // benchmark.report(json_doc);

        } else {
            Serial.printf("WARN: Skipping malformed CSV line: %s\n", line.c_str());
        }
    }
    dataFile.close();

    // --- Final Report Generation ---
    Serial.println("\n--- Run Report ---");
    if (total_inferences > 0) {
        Serial.printf("Total Inferences: %d\n", total_inferences);
        Serial.printf("Average Latency per Inference: %lld us\n", total_latency_us / total_inferences);
        Serial.printf("Peak Internal SRAM Usage: %u bytes\n", peak_internal_sram_usage);
        Serial.printf("Peak PSRAM Usage: %u bytes\n", peak_psram_usage);
    } else {
        Serial.println("No data was processed. Cannot generate report.");
    }
}

/**
 * @brief Parses a comma-separated string into a float array.
 */
bool parse_csv_line(const String& line, float* features, int num_features) {
    int feature_index = 0;
    int current_pos = 0;
    int next_comma = -1;

    for (int i = 0; i < num_features; ++i) {
        if (feature_index >= num_features) return false;

        next_comma = line.indexOf(',', current_pos);

        if (next_comma == -1) {
            if (i != num_features - 1) return false;
            String val_str = line.substring(current_pos);
            if (val_str.length() == 0) return false;
            features[feature_index++] = val_str.toFloat();
            break;
        }

        String val_str = line.substring(current_pos, next_comma);
        if (val_str.length() == 0) return false;
        features[feature_index++] = val_str.toFloat();
        current_pos = next_comma + 1;
    }
    return (feature_index == num_features);
}
