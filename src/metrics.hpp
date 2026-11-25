#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace yolo {

struct LatencyStats {
    size_t count;
    double min_ms;
    double max_ms;
    double mean_ms;
    double std_dev_ms;
    double median_ms;
    double p95_ms;
    double p99_ms;
    double avg_fps;
};

struct ProcessMemory {
    uint64_t peak_rss_kb;
    uint64_t current_rss_kb;
};

int64_t monotonicNowNs();
int64_t processCpuNowNs();

LatencyStats computeStats(const std::vector<int64_t>& samples_ns, std::vector<int64_t>& scratch_sorted);
ProcessMemory readProcessMemory();
uint64_t readExecutableSizeBytes();
double computeCpuUtilizationPct(int64_t cpu_ns, int64_t wall_ns);

} // namespace yolo
