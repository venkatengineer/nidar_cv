const std = @import("std");
const Io = std.Io;

pub const LatencyStats = struct {
    count: usize,
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    std_dev_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    avg_fps: f64,
};

pub const ProcessMemory = struct {
    peak_rss_kb: u64,
    current_rss_kb: u64,
};

/// Computes latency statistics from a slice of durations in nanoseconds.
/// Uses `scratch_sorted` (same length as `samples_ns`) to avoid modifying caller order or allocating.
pub fn computeStats(samples_ns: []const i64, scratch_sorted: []i64) LatencyStats {
    std.debug.assert(samples_ns.len > 0);
    std.debug.assert(scratch_sorted.len == samples_ns.len);

    @memcpy(scratch_sorted, samples_ns);
    std.sort.pdq(i64, scratch_sorted, {}, std.sort.asc(i64));

    var sum_ns: f64 = 0.0;
    for (scratch_sorted) |ns| {
        sum_ns += @floatFromInt(ns);
    }
    const n_f: f64 = @floatFromInt(scratch_sorted.len);
    const mean_ns = sum_ns / n_f;

    var var_sum: f64 = 0.0;
    for (scratch_sorted) |ns| {
        const diff = @as(f64, @floatFromInt(ns)) - mean_ns;
        var_sum += diff * diff;
    }
    const std_dev_ns = @sqrt(var_sum / n_f);

    const ns_to_ms = 1.0 / 1_000_000.0;
    const mean_ms = mean_ns * ns_to_ms;

    return .{
        .count = scratch_sorted.len,
        .min_ms = @as(f64, @floatFromInt(scratch_sorted[0])) * ns_to_ms,
        .max_ms = @as(f64, @floatFromInt(scratch_sorted[scratch_sorted.len - 1])) * ns_to_ms,
        .mean_ms = mean_ms,
        .std_dev_ms = std_dev_ns * ns_to_ms,
        .median_ms = percentileNs(scratch_sorted, 0.50) * ns_to_ms,
        .p95_ms = percentileNs(scratch_sorted, 0.95) * ns_to_ms,
        .p99_ms = percentileNs(scratch_sorted, 0.99) * ns_to_ms,
        .avg_fps = if (mean_ms > 0.0) 1000.0 / mean_ms else 0.0,
    };
}

fn percentileNs(sorted_ns: []const i64, p: f64) f64 {
    if (sorted_ns.len == 1) return @floatFromInt(sorted_ns[0]);
    const pos = p * @as(f64, @floatFromInt(sorted_ns.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi: usize = @min(lo + 1, sorted_ns.len - 1);
    const frac = pos - @as(f64, @floatFromInt(lo));
    const v_lo: f64 = @floatFromInt(sorted_ns[lo]);
    const v_hi: f64 = @floatFromInt(sorted_ns[hi]);
    return v_lo + frac * (v_hi - v_lo);
}

/// Reads Linux /proc/self/status for VmHWM (peak RSS) and VmRSS (current RSS) in KiB,
/// falling back to getrusage(RUSAGE_SELF).maxrss for peak RSS if needed.
pub fn readProcessMemory(io: Io) ProcessMemory {
    var peak_kb: u64 = 0;
    var curr_kb: u64 = 0;

    if (readProcSelfStatus(io)) |buf_res| {
        const content = buf_res.buf[0..buf_res.len];
        peak_kb = parseProcKbLine(content, "VmHWM:") orelse 0;
        curr_kb = parseProcKbLine(content, "VmRSS:") orelse 0;
    }

    if (peak_kb == 0) {
        var ru: std.c.rusage = undefined;
        if (std.c.getrusage(0, &ru) == 0 and ru.maxrss > 0) {
            peak_kb = @intCast(ru.maxrss);
        }
    }

    return .{
        .peak_rss_kb = peak_kb,
        .current_rss_kb = curr_kb,
    };
}

const StatusBuf = struct {
    buf: [8192]u8,
    len: usize,
};

fn readProcSelfStatus(io: Io) ?StatusBuf {
    var file = Io.Dir.openFileAbsolute(io, "/proc/self/status", .{}) catch return null;
    defer file.close(io);

    var res: StatusBuf = .{ .buf = undefined, .len = 0 };
    while (res.len < res.buf.len) {
        const chunk = file.readStreaming(io, &.{res.buf[res.len..]}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return null,
        };
        if (chunk == 0) break;
        res.len += chunk;
    }
    return res;
}

fn parseProcKbLine(content: []const u8, prefix: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, prefix)) {
            const rest = std.mem.trim(u8, line[prefix.len..], " \t\r");
            var it = std.mem.tokenizeAny(u8, rest, " \t");
            const num_str = it.next() orelse return null;
            return std.fmt.parseInt(u64, num_str, 10) catch null;
        }
    }
    return null;
}

/// Returns the byte size of the currently executing binary.
pub fn readExecutableSizeBytes(io: Io) u64 {
    var path_buf: [Io.Dir.max_path_bytes]u8 = undefined;
    const len = std.process.executablePath(io, &path_buf) catch return 0;
    const exe_path = path_buf[0..len];
    const stat = Io.Dir.cwd().statFile(io, exe_path, .{}) catch return 0;
    return stat.size;
}

/// Computes process CPU utilization percentage across all logical cores:
/// (process_cpu_ns / wall_ns) * 100.0 (e.g. 400% means 4 full cores saturated).
pub fn computeCpuUtilizationPct(cpu_ns: i96, wall_ns: i96) f64 {
    if (wall_ns <= 0 or cpu_ns < 0) return 0.0;
    const c_f: f64 = @floatFromInt(cpu_ns);
    const w_f: f64 = @floatFromInt(wall_ns);
    return (c_f / w_f) * 100.0;
}

test "computeStats calculates mean, median, p95, p99, std_dev accurately" {
    var samples: [100]i64 = undefined;
    for (0..100) |i| {
        // 1.00 ms to 100.00 ms
        samples[i] = @intCast((i + 1) * 1_000_000);
    }
    var scratch: [100]i64 = undefined;
    const st = computeStats(&samples, &scratch);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), st.min_ms, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), st.max_ms, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 50.5), st.mean_ms, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 50.5), st.median_ms, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 95.05), st.p95_ms, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 99.01), st.p99_ms, 1e-6);
}
