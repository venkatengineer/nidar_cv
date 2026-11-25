#pragma once

#include "metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace yolo {

struct BenchmarkConfig {
    std::string model_path = "models/yolov26n.onnx";
    std::vector<std::string> image_paths;
    size_t warmup_iters = 100;
    size_t measured_iters = 1000;
    uint32_t threads = 4;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    bool force_nms = false;
    std::string csv_out_path = "benchmarks/cpp_benchmark_results.csv";
    std::string json_out_path = "benchmarks/cpp_benchmark_results.json";
};

struct BenchmarkReport {
    double app_startup_ms;
    double model_load_ms;
    uint64_t binary_size_bytes;
    uint64_t peak_rss_kb;
    uint64_t rss_before_kb;
    uint64_t rss_after_kb;
    double pipeline_cpu_util_pct;
    double infer_only_cpu_util_pct;
    LatencyStats preprocessing;
    LatencyStats pipeline_inference;
    LatencyStats postprocessing;
    LatencyStats total_e2e;
    LatencyStats inference_only;
    size_t last_detections_count;
};

BenchmarkReport runBenchmark(int64_t app_start_ns, const BenchmarkConfig& cfg);

} // namespace yolo
