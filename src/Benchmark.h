#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <Arduino.h>
#include <ArduinoJson.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"

class Benchmark {
public:
    // A structure to hold all the metrics we gather
    struct BenchmarkResult {
        long long inference_latency_us;
        size_t initial_internal_heap;
        size_t final_internal_heap;
        size_t initial_psram;
        size_t final_psram;
        uint32_t cpu_freq_mhz;
    };

    Benchmark() {}

    /**
     * @brief Starts the timer and records the initial memory state.
     */
    void start() {
        // Record initial memory state
        _result.initial_internal_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        _result.initial_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

        // Record start time
        _start_time = esp_timer_get_time();
    }

    /**
     * @brief Stops the timer, records the final memory state, and calculates results.
     */
    void stop() {
        // Record end time and calculate latency
        _end_time = esp_timer_get_time();
        _result.inference_latency_us = _end_time - _start_time;

        // Record final memory state
        _result.final_internal_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        _result.final_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

        // Record CPU frequency
        _result.cpu_freq_mhz = getCpuFrequencyMhz();
    }

    /**
     * @brief Prints the collected benchmark results to the Serial monitor as a JSON string.
     * @param doc A reference to an ArduinoJson StaticJsonDocument to use for serialization.
     */
    void report(JsonDocument& doc) {
        doc.clear();
        doc["inference_latency_us"] = _result.inference_latency_us;

        // Calculate consumed memory, preventing underflow if memory was freed
        size_t internal_ram_consumed = (_result.initial_internal_heap > _result.final_internal_heap)
                                     ? (_result.initial_internal_heap - _result.final_internal_heap)
                                     : 0;
        size_t psram_consumed = (_result.initial_psram > _result.final_psram)
                              ? (_result.initial_psram - _result.final_psram)
                              : 0;

        doc["internal_ram_consumed_b"] = internal_ram_consumed;
        doc["psram_consumed_b"] = psram_consumed;
        doc["cpu_frequency_mhz"] = _result.cpu_freq_mhz;

        serializeJson(doc, Serial);
        Serial.println();
    }

    const BenchmarkResult& getResult() const {
        return _result;
    }

private:
    int64_t _start_time;
    int64_t _end_time;
    BenchmarkResult _result;
};

#endif // BENCHMARK_H
