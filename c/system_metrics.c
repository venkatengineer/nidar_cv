#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

int64_t sys_monotonic_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

int64_t sys_process_cpu_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

void sys_read_process_memory(int64_t* out_peak_rss_kb, int64_t* out_curr_rss_kb) {
    int64_t peak_kb = 0;
    int64_t curr_kb = 0;

    FILE* fp = fopen("/proc/self/status", "r");
    if (fp != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), fp) != NULL) {
            unsigned long long val = 0;
            if (strncmp(line, "VmHWM:", 6) == 0) {
                if (sscanf(line + 6, "%llu", &val) == 1) peak_kb = (int64_t)val;
            } else if (strncmp(line, "VmRSS:", 6) == 0) {
                if (sscanf(line + 6, "%llu", &val) == 1) curr_kb = (int64_t)val;
            }
        }
        fclose(fp);
    }

    if (peak_kb == 0) {
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
            peak_kb = (int64_t)ru.ru_maxrss;
        }
    }

    *out_peak_rss_kb = peak_kb;
    *out_curr_rss_kb = curr_kb;
}

int64_t sys_executable_size_bytes(void) {
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len <= 0) return 0;
    exe_path[len] = '\0';

    struct stat st;
    if (stat(exe_path, &st) != 0) return 0;
    return (int64_t)st.st_size;
}
