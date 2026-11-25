#include "benchmark.hpp"

#include "detector.hpp"
#include "image.hpp"
#include "postprocessing.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace yolo {

namespace {

std::string compilerVersionString() {
#if defined(__clang__)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "clang++ %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
    return buf;
#elif defined(__GNUC__)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "g++ %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    return buf;
#else
    return "unknown";
#endif
}

const char* optimizeModeString() {
#ifdef NDEBUG
    return "release";
#else
    return "debug";
#endif
}

void printStatsRow(const char* label, const LatencyStats& s) {
    std::printf(
        "%-22s | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.2f\n",
        label,
        s.mean_ms,
        s.median_ms,
        s.p95_ms,
        s.p99_ms,
        s.min_ms,
        s.max_ms,
        s.avg_fps);
}

void printReport(const Detector& det, const BenchmarkConfig& cfg, const BenchmarkReport& r) {
    const auto& in_shape = det.inputShape();
    const auto& out_shape = det.outputShape();

    std::printf("\n=====================================================================\n");
    std::printf(" C++17 Edge Computer Vision Benchmark — YOLO26n (ONNX Runtime CPU)\n");
    std::printf("=====================================================================\n");
    std::printf("Compiler            : %s (optimize=%s)\n", compilerVersionString().c_str(), optimizeModeString());
    std::printf("ONNX Runtime        : %s (CPUExecutionProvider, threads=%u)\n", det.ortVersion().c_str(), cfg.threads);
    std::printf("Model               : %s\n", cfg.model_path.c_str());
    std::printf("Input Tensor        : [%lld, %lld, %lld, %lld] (FP32)\n",
                static_cast<long long>(in_shape[0]),
                static_cast<long long>(in_shape[1]),
                static_cast<long long>(in_shape[2]),
                static_cast<long long>(in_shape[3]));
    std::printf("Output Tensor       : [%lld, %lld, %lld] (%s)\n",
                static_cast<long long>(out_shape[0]),
                static_cast<long long>(out_shape[1]),
                static_cast<long long>(out_shape[2]),
                outputFormatName(det.outputDesc().format));
    std::printf("Iterations          : %zu warmup + %zu measured (%zu fixed image(s))\n",
                cfg.warmup_iters, cfg.measured_iters, cfg.image_paths.size());
    std::printf("---------------------------------------------------------------------\n");
    std::printf("Startup & Resources:\n");
    std::printf("  App Startup Time  : %.3f ms\n", r.app_startup_ms);
    std::printf("  Model Load Time   : %.3f ms\n", r.model_load_ms);
    std::printf("  Binary Size       : %llu bytes (%.2f KiB)\n",
                static_cast<unsigned long long>(r.binary_size_bytes),
                static_cast<double>(r.binary_size_bytes) / 1024.0);
    std::printf("  Peak RSS (VmHWM)  : %llu KiB (%.2f MiB)\n",
                static_cast<unsigned long long>(r.peak_rss_kb),
                static_cast<double>(r.peak_rss_kb) / 1024.0);
    std::printf("  RSS Before/After  : %llu KiB -> %llu KiB (delta=%lld KiB)\n",
                static_cast<unsigned long long>(r.rss_before_kb),
                static_cast<unsigned long long>(r.rss_after_kb),
                static_cast<long long>(r.rss_after_kb) - static_cast<long long>(r.rss_before_kb));
    std::printf("  CPU Utilization   : %.2f%% (pipeline), %.2f%% (inference-only) [process CPU/wall]\n",
                r.pipeline_cpu_util_pct, r.infer_only_cpu_util_pct);
    std::printf("---------------------------------------------------------------------\n");
    std::printf("Stage Latencies (ms)   |    Mean |  Median |     P95 |     P99 |     Min |     Max |     FPS\n");
    printStatsRow("Preprocessing", r.preprocessing);
    printStatsRow("Inference (Pipeline)", r.pipeline_inference);
    printStatsRow("Postprocessing", r.postprocessing);
    printStatsRow("Total End-to-End", r.total_e2e);
    printStatsRow("Inference-Only (Isol)", r.inference_only);
    std::printf("=====================================================================\n\n");
}

void writeCsvReport(const Detector& det, const BenchmarkConfig& cfg, const BenchmarkReport& r) {
    std::ofstream ofs(cfg.csv_out_path, std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open CSV output: " + cfg.csv_out_path);
    }

    ofs << "language,compiler_version,optimize_mode,ort_version,threads,warmup_iters,measured_iters,stage,count,"
        << "mean_ms,median_ms,p95_ms,p99_ms,min_ms,max_ms,std_dev_ms,avg_fps,"
        << "model_load_ms,app_startup_ms,peak_rss_kb,cpu_util_pct,binary_size_bytes\n";

    struct Row {
        const char* name;
        const LatencyStats& stats;
        double cpu;
    };
    const Row rows[] = {
        {"preprocessing", r.preprocessing, r.pipeline_cpu_util_pct},
        {"pipeline_inference", r.pipeline_inference, r.pipeline_cpu_util_pct},
        {"postprocessing", r.postprocessing, r.pipeline_cpu_util_pct},
        {"total_e2e", r.total_e2e, r.pipeline_cpu_util_pct},
        {"inference_only", r.inference_only, r.infer_only_cpu_util_pct},
    };

    const std::string comp = compilerVersionString();
    const char* opt_mode = optimizeModeString();

    for (const auto& row : rows) {
        ofs << std::fixed << std::setprecision(6)
            << "cpp," << comp << "," << opt_mode << "," << det.ortVersion() << ","
            << cfg.threads << "," << cfg.warmup_iters << "," << cfg.measured_iters << ","
            << row.name << "," << row.stats.count << ","
            << row.stats.mean_ms << "," << row.stats.median_ms << ","
            << row.stats.p95_ms << "," << row.stats.p99_ms << ","
            << row.stats.min_ms << "," << row.stats.max_ms << ","
            << row.stats.std_dev_ms << ","
            << std::setprecision(4) << row.stats.avg_fps << ","
            << std::setprecision(6) << r.model_load_ms << "," << r.app_startup_ms << ","
            << r.peak_rss_kb << ","
            << std::setprecision(4) << row.cpu << ","
            << r.binary_size_bytes << "\n";
    }
}

void writeJsonStats(std::ofstream& ofs, const char* name, const LatencyStats& s, bool trailing_comma) {
    ofs << std::fixed
        << "    \"" << name << "\": {\n"
        << "      \"count\": " << s.count << ",\n"
        << std::setprecision(6)
        << "      \"mean_ms\": " << s.mean_ms << ",\n"
        << "      \"median_ms\": " << s.median_ms << ",\n"
        << "      \"p95_ms\": " << s.p95_ms << ",\n"
        << "      \"p99_ms\": " << s.p99_ms << ",\n"
        << "      \"min_ms\": " << s.min_ms << ",\n"
        << "      \"max_ms\": " << s.max_ms << ",\n"
        << "      \"std_dev_ms\": " << s.std_dev_ms << ",\n"
        << std::setprecision(4)
        << "      \"avg_fps\": " << s.avg_fps << "\n"
        << "    }" << (trailing_comma ? "," : "") << "\n";
}

void writeJsonReport(const Detector& det, const BenchmarkConfig& cfg, const BenchmarkReport& r) {
    std::ofstream ofs(cfg.json_out_path, std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open JSON output: " + cfg.json_out_path);
    }

    const auto& in_shape = det.inputShape();
    const auto& out_shape = det.outputShape();

    ofs << std::fixed
        << "{\n"
        << "  \"language\": \"cpp\",\n"
        << "  \"compiler_version\": \"" << compilerVersionString() << "\",\n"
        << "  \"optimize_mode\": \"" << optimizeModeString() << "\",\n"
        << "  \"onnxruntime_version\": \"" << det.ortVersion() << "\",\n"
        << "  \"execution_provider\": \"CPUExecutionProvider\",\n"
        << "  \"model_path\": \"" << cfg.model_path << "\",\n"
        << "  \"input_shape\": [" << in_shape[0] << ", " << in_shape[1] << ", " << in_shape[2] << ", " << in_shape[3] << "],\n"
        << "  \"output_shape\": [" << out_shape[0] << ", " << out_shape[1] << ", " << out_shape[2] << "],\n"
        << "  \"output_format\": \"" << outputFormatName(det.outputDesc().format) << "\",\n"
        << "  \"config\": {\n"
        << "    \"threads\": " << cfg.threads << ",\n"
        << "    \"warmup_iterations\": " << cfg.warmup_iters << ",\n"
        << "    \"measured_iterations\": " << cfg.measured_iters << ",\n"
        << std::setprecision(4)
        << "    \"conf_threshold\": " << cfg.conf_threshold << ",\n"
        << "    \"iou_threshold\": " << cfg.iou_threshold << ",\n"
        << "    \"image_count\": " << cfg.image_paths.size() << "\n"
        << "  },\n"
        << "  \"system_metrics\": {\n"
        << std::setprecision(6)
        << "    \"app_startup_ms\": " << r.app_startup_ms << ",\n"
        << "    \"model_load_ms\": " << r.model_load_ms << ",\n"
        << "    \"binary_size_bytes\": " << r.binary_size_bytes << ",\n"
        << "    \"peak_rss_kb\": " << r.peak_rss_kb << ",\n"
        << "    \"rss_before_kb\": " << r.rss_before_kb << ",\n"
        << "    \"rss_after_kb\": " << r.rss_after_kb << ",\n"
        << std::setprecision(4)
        << "    \"pipeline_cpu_utilization_pct\": " << r.pipeline_cpu_util_pct << ",\n"
        << "    \"inference_only_cpu_utilization_pct\": " << r.infer_only_cpu_util_pct << "\n"
        << "  },\n"
        << "  \"stages\": {\n";

    writeJsonStats(ofs, "preprocessing", r.preprocessing, true);
    writeJsonStats(ofs, "pipeline_inference", r.pipeline_inference, true);
    writeJsonStats(ofs, "postprocessing", r.postprocessing, true);
    writeJsonStats(ofs, "total_end_to_end", r.total_e2e, true);
    writeJsonStats(ofs, "inference_only", r.inference_only, false);

    ofs << "  }\n}\n";
}

} // namespace

BenchmarkReport runBenchmark(int64_t app_start_ns, const BenchmarkConfig& cfg) {
    if (cfg.image_paths.empty()) {
        throw std::runtime_error("No input images provided for benchmark");
    }

    std::vector<Image> images;
    images.reserve(cfg.image_paths.size());
    for (const auto& path : cfg.image_paths) {
        images.push_back(Image::load(path));
    }

    const int64_t load_start = monotonicNowNs();
    Detector det(DetectorConfig{
        cfg.model_path,
        cfg.threads,
        1,
        cfg.conf_threshold,
        cfg.iou_threshold,
        cfg.force_nms,
    });
    const int64_t load_end = monotonicNowNs();
    const int64_t ready_ns = monotonicNowNs();

    const double model_load_ms = static_cast<double>(load_end - load_start) / 1'000'000.0;
    const double app_startup_ms = static_cast<double>(ready_ns - app_start_ns) / 1'000'000.0;

    std::vector<int64_t> pre_ns(cfg.measured_iters);
    std::vector<int64_t> inf_ns(cfg.measured_iters);
    std::vector<int64_t> post_ns(cfg.measured_iters);
    std::vector<int64_t> tot_ns(cfg.measured_iters);
    std::vector<int64_t> inf_only_ns(cfg.measured_iters);
    std::vector<int64_t> scratch_ns(cfg.measured_iters);

    const ProcessMemory rss_before = readProcessMemory();

    // Pass 1: Complete End-to-End Pipeline
    for (size_t i = 0; i < cfg.warmup_iters; ++i) {
        const Image& img = images[i % images.size()];
        const LetterboxParams lb = det.preprocess(img.data(), img.width(), img.height());
        det.inferRaw();
        const size_t n = det.postprocess(lb);
        asm volatile("" : : "r"(n) : "memory");
    }

    const int64_t pipe_wall_start = monotonicNowNs();
    const int64_t pipe_cpu_start = processCpuNowNs();

    size_t last_det_count = 0;
    for (size_t i = 0; i < cfg.measured_iters; ++i) {
        const Image& img = images[i % images.size()];

        const int64_t t0 = monotonicNowNs();
        const LetterboxParams lb = det.preprocess(img.data(), img.width(), img.height());
        const int64_t t1 = monotonicNowNs();
        det.inferRaw();
        const int64_t t2 = monotonicNowNs();
        const size_t n = det.postprocess(lb);
        const int64_t t3 = monotonicNowNs();

        asm volatile("" : : "r"(n) : "memory");
        last_det_count = n;

        pre_ns[i] = t1 - t0;
        inf_ns[i] = t2 - t1;
        post_ns[i] = t3 - t2;
        tot_ns[i] = t3 - t0;
    }

    const int64_t pipe_cpu_end = processCpuNowNs();
    const int64_t pipe_wall_end = monotonicNowNs();

    // Pass 2: Inference-Only Benchmark
    (void)det.preprocess(images[0].data(), images[0].width(), images[0].height());
    for (size_t i = 0; i < cfg.warmup_iters; ++i) {
        det.inferRaw();
    }

    const int64_t inf_wall_start = monotonicNowNs();
    const int64_t inf_cpu_start = processCpuNowNs();

    for (size_t i = 0; i < cfg.measured_iters; ++i) {
        const int64_t t0 = monotonicNowNs();
        det.inferRaw();
        const int64_t t1 = monotonicNowNs();
        inf_only_ns[i] = t1 - t0;
    }

    const int64_t inf_cpu_end = processCpuNowNs();
    const int64_t inf_wall_end = monotonicNowNs();

    const ProcessMemory rss_after = readProcessMemory();
    const uint64_t bin_size = readExecutableSizeBytes();

    const BenchmarkReport report{
        app_startup_ms,
        model_load_ms,
        bin_size,
        rss_after.peak_rss_kb,
        rss_before.current_rss_kb,
        rss_after.current_rss_kb,
        computeCpuUtilizationPct(pipe_cpu_end - pipe_cpu_start, pipe_wall_end - pipe_wall_start),
        computeCpuUtilizationPct(inf_cpu_end - inf_cpu_start, inf_wall_end - inf_wall_start),
        computeStats(pre_ns, scratch_ns),
        computeStats(inf_ns, scratch_ns),
        computeStats(post_ns, scratch_ns),
        computeStats(tot_ns, scratch_ns),
        computeStats(inf_only_ns, scratch_ns),
        last_det_count,
    };

    printReport(det, cfg, report);
    writeCsvReport(det, cfg, report);
    writeJsonReport(det, cfg, report);

    return report;
}

} // namespace yolo
