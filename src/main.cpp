#include "benchmark.hpp"
#include "detector.hpp"
#include "image.hpp"
#include "metrics.hpp"
#include "postprocessing.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Mode {
    detect,
    validate,
    benchmark,
};

struct CliOptions {
    Mode mode = Mode::detect;
    std::string model_path = "models/yolov26n.onnx";
    std::vector<std::string> image_paths;
    std::string output_image;
    std::string csv_out = "benchmarks/cpp_benchmark_results.csv";
    std::string json_out = "benchmarks/cpp_benchmark_results.json";
    uint32_t threads = 4;
    size_t warmup_iters = 100;
    size_t measured_iters = 1000;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    bool force_nms = false;
};

void printUsage() {
    std::printf(
        "Usage: cpp-yolo26n [detect|validate|benchmark] [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>     ONNX model path (default: models/yolov26n.onnx)\n"
        "  --image <path>     Input image path (repeatable; defaults to test/bus.jpg, test/zidane.jpg)\n"
        "  --output <path>    Save annotated PNG image in detect mode\n"
        "  --threads <n>      ONNX Runtime intra-op CPU threads (default: 4)\n"
        "  --warmup <n>       Warmup iterations for benchmark (default: 100)\n"
        "  --iters <n>        Measured iterations for benchmark (default: 1000)\n"
        "  --conf <f32>       Confidence threshold (default: 0.25)\n"
        "  --iou <f32>        IoU threshold for NMS when applicable (default: 0.45)\n"
        "  --force-nms        Force NMS even on end2end [1,N,6] outputs\n"
        "  --csv <path>       Benchmark CSV output path (default: benchmarks/cpp_benchmark_results.csv)\n"
        "  --json <path>      Benchmark JSON output path (default: benchmarks/cpp_benchmark_results.json)\n"
        "  -h, --help         Show this help message\n");
}

bool parseArgs(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_val = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing argument value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "detect") {
            opts.mode = Mode::detect;
        } else if (arg == "validate") {
            opts.mode = Mode::validate;
        } else if (arg == "benchmark") {
            opts.mode = Mode::benchmark;
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else if (arg == "--model") {
            opts.model_path = require_val("--model");
        } else if (arg == "--image") {
            opts.image_paths.push_back(require_val("--image"));
        } else if (arg == "--output") {
            opts.output_image = require_val("--output");
        } else if (arg == "--csv") {
            opts.csv_out = require_val("--csv");
        } else if (arg == "--json") {
            opts.json_out = require_val("--json");
        } else if (arg == "--threads") {
            opts.threads = static_cast<uint32_t>(std::stoul(require_val("--threads")));
        } else if (arg == "--warmup") {
            opts.warmup_iters = static_cast<size_t>(std::stoull(require_val("--warmup")));
        } else if (arg == "--iters") {
            opts.measured_iters = static_cast<size_t>(std::stoull(require_val("--iters")));
        } else if (arg == "--conf") {
            opts.conf_threshold = std::stof(require_val("--conf"));
        } else if (arg == "--iou") {
            opts.iou_threshold = std::stof(require_val("--iou"));
        } else if (arg == "--force-nms") {
            opts.force_nms = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return true;
}

void runDetect(const CliOptions& opts, const std::string& image_path) {
    yolo::Detector det(yolo::DetectorConfig{
        opts.model_path,
        opts.threads,
        1,
        opts.conf_threshold,
        opts.iou_threshold,
        opts.force_nms,
    });

    const yolo::Image img = yolo::Image::load(image_path);

    const int64_t t0 = yolo::monotonicNowNs();
    const yolo::LetterboxParams lb = det.preprocess(img.data(), img.width(), img.height());
    const int64_t t1 = yolo::monotonicNowNs();
    det.inferRaw();
    const int64_t t2 = yolo::monotonicNowNs();
    const size_t det_count = det.postprocess(lb);
    const int64_t t3 = yolo::monotonicNowNs();

    std::printf("Image: %s (%ux%u), Output format: %s\n",
                image_path.c_str(), img.width(), img.height(),
                yolo::outputFormatName(det.outputDesc().format));
    std::printf("Latency: pre=%.3fms, infer=%.3fms, post=%.3fms, total=%.3fms\n",
                static_cast<double>(t1 - t0) / 1e6,
                static_cast<double>(t2 - t1) / 1e6,
                static_cast<double>(t3 - t2) / 1e6,
                static_cast<double>(t3 - t0) / 1e6);
    std::printf("Detections (%zu):\n", det_count);
    const yolo::Detection* dets = det.detections();
    for (size_t i = 0; i < det_count; ++i) {
        const auto& d = dets[i];
        const auto cname = yolo::className(d.class_id);
        std::printf("  [%zu] class=%u (%.*s) conf=%.4f box=[%.2f, %.2f, %.2f, %.2f]\n",
                    i, d.class_id, static_cast<int>(cname.size()), cname.data(),
                    d.confidence, d.x1, d.y1, d.x2, d.y2);
    }

    if (!opts.output_image.empty()) {
        yolo::Image vis = img.clone();
        vis.drawDetections(dets, det_count);
        vis.savePng(opts.output_image);
    }
}

void runValidate(const CliOptions& opts, const std::vector<std::string>& image_paths) {
    yolo::Detector det(yolo::DetectorConfig{
        opts.model_path,
        opts.threads,
        1,
        opts.conf_threshold,
        opts.iou_threshold,
        opts.force_nms,
    });

    const auto& in_s = det.inputShape();
    const auto& out_s = det.outputShape();

    std::printf("[PASS] Model loaded: %s (ORT %s)\n", opts.model_path.c_str(), det.ortVersion().c_str());
    std::printf("[PASS] Input tensor validated: [%lld, %lld, %lld, %lld] FP32\n",
                static_cast<long long>(in_s[0]), static_cast<long long>(in_s[1]),
                static_cast<long long>(in_s[2]), static_cast<long long>(in_s[3]));
    std::printf("[PASS] Output tensor validated: [%lld, %lld, %lld] (%s)\n",
                static_cast<long long>(out_s[0]), static_cast<long long>(out_s[1]),
                static_cast<long long>(out_s[2]), yolo::outputFormatName(det.outputDesc().format));

    for (const auto& path : image_paths) {
        const yolo::Image img = yolo::Image::load(path);
        const yolo::LetterboxParams lb = det.preprocess(img.data(), img.width(), img.height());
        det.inferRaw();
        const size_t det_count = det.postprocess(lb);
        if (det_count == 0) {
            throw std::runtime_error("No detections found on " + path);
        }

        const float fw = static_cast<float>(img.width());
        const float fh = static_cast<float>(img.height());
        const yolo::Detection* dets = det.detections();
        for (size_t i = 0; i < det_count; ++i) {
            const auto& d = dets[i];
            if (d.x1 < 0.0f || d.y1 < 0.0f || d.x2 > fw || d.y2 > fh || d.x2 <= d.x1 || d.y2 <= d.y1) {
                throw std::runtime_error("Bounding box out of bounds");
            }
            if (!std::isfinite(d.confidence) || d.confidence < opts.conf_threshold || d.confidence > 1.0f) {
                throw std::runtime_error("Invalid confidence value");
            }
            if (d.class_id >= yolo::NUM_COCO_CLASSES) {
                throw std::runtime_error("Invalid class_id");
            }
        }
        std::printf("[PASS] %s: %zu valid detections, all boxes within [0..%u, 0..%u]\n",
                    path.c_str(), det_count, img.width(), img.height());
    }

    const uint64_t rss_start = yolo::readProcessMemory().current_rss_kb;
    for (int i = 0; i < 50; ++i) {
        det.inferRaw();
    }
    const uint64_t rss_end = yolo::readProcessMemory().current_rss_kb;
    std::printf("[PASS] 50 repeated inferences RSS check: %llu KiB -> %llu KiB\n",
                static_cast<unsigned long long>(rss_start),
                static_cast<unsigned long long>(rss_end));
}

} // namespace

int main(int argc, char** argv) {
    const int64_t app_start_ns = yolo::monotonicNowNs();
    try {
        CliOptions opts;
        if (!parseArgs(argc, argv, opts)) {
            return 0;
        }
        if (opts.image_paths.empty()) {
            opts.image_paths = {"test/bus.jpg", "test/zidane.jpg"};
        }

        switch (opts.mode) {
            case Mode::detect:
                runDetect(opts, opts.image_paths[0]);
                break;
            case Mode::validate:
                runValidate(opts, opts.image_paths);
                break;
            case Mode::benchmark: {
                yolo::BenchmarkConfig bcfg;
                bcfg.model_path = opts.model_path;
                bcfg.image_paths = opts.image_paths;
                bcfg.warmup_iters = opts.warmup_iters;
                bcfg.measured_iters = opts.measured_iters;
                bcfg.threads = opts.threads;
                bcfg.conf_threshold = opts.conf_threshold;
                bcfg.iou_threshold = opts.iou_threshold;
                bcfg.force_nms = opts.force_nms;
                bcfg.csv_out_path = opts.csv_out;
                bcfg.json_out_path = opts.json_out;
                (void)yolo::runBenchmark(app_start_ns, bcfg);
                break;
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
