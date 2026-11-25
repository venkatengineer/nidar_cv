#include "metrics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace yolo {

namespace {

double percentileNs(const std::vector<int64_t>& sorted_ns, double p) {
    if (sorted_ns.size() == 1) return static_cast<double>(sorted_ns[0]);
    const double pos = p * static_cast<double>(sorted_ns.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted_ns.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    const double v_lo = static_cast<double>(sorted_ns[lo]);
    const double v_hi = static_cast<double>(sorted_ns[hi]);
    return v_lo + frac * (v_hi - v_lo);
}

uint64_t parseProcKbLine(const std::string& content, const char* prefix) {
    const size_t prefix_len = std::strlen(prefix);
    size_t pos = 0;
    while (pos < content.size()) {
        size_t nl = content.find('\n', pos);
        if (nl == std::string::npos) nl = content.size();
        if (nl - pos >= prefix_len && content.compare(pos, prefix_len, prefix) == 0) {
            uint64_t val = 0;
            if (std::sscanf(content.c_str() + pos + prefix_len, "%llu",
                            reinterpret_cast<unsigned long long*>(&val)) == 1) {
                return val;
            }
        }
        pos = nl + 1;
    }
    return 0;
}

} // namespace

int64_t monotonicNowNs() {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

int64_t processCpuNowNs() {
    struct timespec ts {};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

LatencyStats computeStats(const std::vector<int64_t>& samples_ns, std::vector<int64_t>& scratch_sorted) {
    assert(!samples_ns.empty());
    assert(scratch_sorted.size() == samples_ns.size());

    std::copy(samples_ns.begin(), samples_ns.end(), scratch_sorted.begin());
    std::sort(scratch_sorted.begin(), scratch_sorted.end());

    double sum_ns = 0.0;
    for (int64_t ns : scratch_sorted) {
        sum_ns += static_cast<double>(ns);
    }
    const double n_f = static_cast<double>(scratch_sorted.size());
    const double mean_ns = sum_ns / n_f;

    double var_sum = 0.0;
    for (int64_t ns : scratch_sorted) {
        const double diff = static_cast<double>(ns) - mean_ns;
        var_sum += diff * diff;
    }
    const double std_dev_ns = std::sqrt(var_sum / n_f);

    constexpr double ns_to_ms = 1.0 / 1'000'000.0;
    const double mean_ms = mean_ns * ns_to_ms;

    return LatencyStats{
        scratch_sorted.size(),
        static_cast<double>(scratch_sorted.front()) * ns_to_ms,
        static_cast<double>(scratch_sorted.back()) * ns_to_ms,
        mean_ms,
        std_dev_ns * ns_to_ms,
        percentileNs(scratch_sorted, 0.50) * ns_to_ms,
        percentileNs(scratch_sorted, 0.95) * ns_to_ms,
        percentileNs(scratch_sorted, 0.99) * ns_to_ms,
        mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0,
    };
}

ProcessMemory readProcessMemory() {
    uint64_t peak_kb = 0;
    uint64_t curr_kb = 0;

    std::ifstream ifs("/proc/self/status");
    if (ifs) {
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        peak_kb = parseProcKbLine(content, "VmHWM:");
        curr_kb = parseProcKbLine(content, "VmRSS:");
    }

    if (peak_kb == 0) {
        struct rusage ru {};
        if (getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
            peak_kb = static_cast<uint64_t>(ru.ru_maxrss);
        }
    }

    return ProcessMemory{peak_kb, curr_kb};
}

uint64_t readExecutableSizeBytes() {
    char exe_path[4096];
    const ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len <= 0) return 0;
    exe_path[len] = '\0';

    struct stat st {};
    if (stat(exe_path, &st) != 0) return 0;
    return static_cast<uint64_t>(st.st_size);
}

double computeCpuUtilizationPct(int64_t cpu_ns, int64_t wall_ns) {
    if (wall_ns <= 0 || cpu_ns < 0) return 0.0;
    return (static_cast<double>(cpu_ns) / static_cast<double>(wall_ns)) * 100.0;
}

} // namespace yolo
