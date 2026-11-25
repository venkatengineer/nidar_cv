const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Clock = Io.Clock;

const image_mod = @import("image.zig");
const detector_mod = @import("detector.zig");
const postprocessing = @import("postprocessing.zig");
const metrics = @import("metrics.zig");

pub const BenchmarkConfig = struct {
    model_path: []const u8,
    image_paths: []const []const u8,
    warmup_iters: usize = 100,
    measured_iters: usize = 1000,
    threads: u32 = 4,
    conf_threshold: f32 = 0.25,
    iou_threshold: f32 = 0.45,
    force_nms: bool = false,
    csv_out_path: []const u8 = "benchmarks/zig_benchmark_results.csv",
    json_out_path: []const u8 = "benchmarks/zig_benchmark_results.json",
};

pub const BenchmarkReport = struct {
    app_startup_ms: f64,
    model_load_ms: f64,
    binary_size_bytes: u64,
    peak_rss_kb: u64,
    rss_before_kb: u64,
    rss_after_kb: u64,
    pipeline_cpu_util_pct: f64,
    infer_only_cpu_util_pct: f64,
    preprocessing: metrics.LatencyStats,
    pipeline_inference: metrics.LatencyStats,
    postprocessing: metrics.LatencyStats,
    total_e2e: metrics.LatencyStats,
    inference_only: metrics.LatencyStats,
    last_detections_count: usize,
};

pub fn runBenchmark(
    io: Io,
    allocator: std.mem.Allocator,
    app_start_ts: Clock.Timestamp,
    cfg: BenchmarkConfig,
) !BenchmarkReport {
    if (cfg.image_paths.len == 0) return error.NoInputImages;

    // Preload fixed test images into memory once before benchmarking
    // (benchmark measures preprocessing + inference + postprocessing, not disk I/O jitter,
    // while keeping identical decoded RGB buffers across language comparisons).
    var images = try allocator.alloc(image_mod.Image, cfg.image_paths.len);
    var loaded_count: usize = 0;
    defer {
        for (images[0..loaded_count]) |*img| {
            img.deinit();
        }
        allocator.free(images);
    }

    for (cfg.image_paths, 0..) |path, idx| {
        images[idx] = try image_mod.Image.load(allocator, path);
        loaded_count += 1;
    }

    const load_start = Clock.Timestamp.now(io, .awake);
    var det = try detector_mod.Detector.init(allocator, .{
        .model_path = cfg.model_path,
        .intra_op_threads = cfg.threads,
        .inter_op_threads = 1,
        .conf_threshold = cfg.conf_threshold,
        .iou_threshold = cfg.iou_threshold,
        .force_nms = cfg.force_nms,
    });
    defer det.deinit();
    const load_end = Clock.Timestamp.now(io, .awake);

    const model_load_ms = @as(f64, @floatFromInt(load_start.durationTo(load_end).raw.nanoseconds)) / 1_000_000.0;
    const ready_ts = Clock.Timestamp.now(io, .awake);
    const app_startup_ms = @as(f64, @floatFromInt(app_start_ts.durationTo(ready_ts).raw.nanoseconds)) / 1_000_000.0;

    // Preallocate latency sample arrays once
    const pre_ns = try allocator.alloc(i64, cfg.measured_iters);
    defer allocator.free(pre_ns);
    const inf_ns = try allocator.alloc(i64, cfg.measured_iters);
    defer allocator.free(inf_ns);
    const post_ns = try allocator.alloc(i64, cfg.measured_iters);
    defer allocator.free(post_ns);
    const tot_ns = try allocator.alloc(i64, cfg.measured_iters);
    defer allocator.free(tot_ns);
    const inf_only_ns = try allocator.alloc(i64, cfg.measured_iters);
    defer allocator.free(inf_only_ns);
    const scratch_ns = try allocator.alloc(i64, cfg.measured_iters);
    defer allocator.free(scratch_ns);

    const rss_before = metrics.readProcessMemory(io);

    // =========================================================================
    // Pass 1: Complete End-to-End Pipeline (Preprocess + Inference + Postprocess)
    // =========================================================================
    for (0..cfg.warmup_iters) |i| {
        const img = &images[i % images.len];
        const lb = det.preprocess(img.data, img.width, img.height);
        try det.inferRaw();
        const res = det.postprocess(lb);
        std.mem.doNotOptimizeAway(res.len);
    }

    const pipe_wall_start = Clock.Timestamp.now(io, .awake);
    const pipe_cpu_start = Clock.Timestamp.now(io, .cpu_process);

    var last_det_count: usize = 0;
    for (0..cfg.measured_iters) |i| {
        const img = &images[i % images.len];

        const t0 = Clock.Timestamp.now(io, .awake);
        const lb = det.preprocess(img.data, img.width, img.height);
        const t1 = Clock.Timestamp.now(io, .awake);
        try det.inferRaw();
        const t2 = Clock.Timestamp.now(io, .awake);
        const res = det.postprocess(lb);
        const t3 = Clock.Timestamp.now(io, .awake);

        std.mem.doNotOptimizeAway(res.len);
        last_det_count = res.len;

        pre_ns[i] = @intCast(t0.durationTo(t1).raw.nanoseconds);
        inf_ns[i] = @intCast(t1.durationTo(t2).raw.nanoseconds);
        post_ns[i] = @intCast(t2.durationTo(t3).raw.nanoseconds);
        tot_ns[i] = @intCast(t0.durationTo(t3).raw.nanoseconds);
    }

    const pipe_cpu_end = Clock.Timestamp.now(io, .cpu_process);
    const pipe_wall_end = Clock.Timestamp.now(io, .awake);

    // =========================================================================
    // Pass 2: Inference-Only Benchmark (ONNX Runtime Session::Run isolated)
    // =========================================================================
    _ = det.preprocess(images[0].data, images[0].width, images[0].height);
    for (0..cfg.warmup_iters) |_| {
        try det.inferRaw();
    }

    const inf_wall_start = Clock.Timestamp.now(io, .awake);
    const inf_cpu_start = Clock.Timestamp.now(io, .cpu_process);

    for (0..cfg.measured_iters) |i| {
        const t0 = Clock.Timestamp.now(io, .awake);
        try det.inferRaw();
        const t1 = Clock.Timestamp.now(io, .awake);
        inf_only_ns[i] = @intCast(t0.durationTo(t1).raw.nanoseconds);
    }

    const inf_cpu_end = Clock.Timestamp.now(io, .cpu_process);
    const inf_wall_end = Clock.Timestamp.now(io, .awake);

    const rss_after = metrics.readProcessMemory(io);
    const bin_size = metrics.readExecutableSizeBytes(io);

    const report = BenchmarkReport{
        .app_startup_ms = app_startup_ms,
        .model_load_ms = model_load_ms,
        .binary_size_bytes = bin_size,
        .peak_rss_kb = rss_after.peak_rss_kb,
        .rss_before_kb = rss_before.current_rss_kb,
        .rss_after_kb = rss_after.current_rss_kb,
        .pipeline_cpu_util_pct = metrics.computeCpuUtilizationPct(
            pipe_cpu_start.durationTo(pipe_cpu_end).raw.nanoseconds,
            pipe_wall_start.durationTo(pipe_wall_end).raw.nanoseconds,
        ),
        .infer_only_cpu_util_pct = metrics.computeCpuUtilizationPct(
            inf_cpu_start.durationTo(inf_cpu_end).raw.nanoseconds,
            inf_wall_start.durationTo(inf_wall_end).raw.nanoseconds,
        ),
        .preprocessing = metrics.computeStats(pre_ns, scratch_ns),
        .pipeline_inference = metrics.computeStats(inf_ns, scratch_ns),
        .postprocessing = metrics.computeStats(post_ns, scratch_ns),
        .total_e2e = metrics.computeStats(tot_ns, scratch_ns),
        .inference_only = metrics.computeStats(inf_only_ns, scratch_ns),
        .last_detections_count = last_det_count,
    };

    try printReport(io, &det, cfg, report);
    try writeCsvReport(io, allocator, &det, cfg, report);
    try writeJsonReport(io, allocator, &det, cfg, report);

    return report;
}

fn printReport(
    io: Io,
    det: *const detector_mod.Detector,
    cfg: BenchmarkConfig,
    r: BenchmarkReport,
) !void {
    var out_buf: [4096]u8 = undefined;
    var fw = Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &fw.interface;

    try w.print("\n=====================================================================\n", .{});
    try w.print(" Zig Edge Computer Vision Benchmark — YOLO26n (ONNX Runtime CPU)\n", .{});
    try w.print("=====================================================================\n", .{});
    try w.print("Zig Version         : {s} (optimize={s})\n", .{ builtin.zig_version_string, @tagName(builtin.mode) });
    try w.print("ONNX Runtime        : {s} (CPUExecutionProvider, threads={d})\n", .{ det.ort_version, cfg.threads });
    try w.print("Model               : {s}\n", .{cfg.model_path});
    try w.print("Input Tensor        : [{d}, {d}, {d}, {d}] (FP32)\n", .{
        det.input_shape[0], det.input_shape[1], det.input_shape[2], det.input_shape[3],
    });
    try w.print("Output Tensor       : [{d}, {d}, {d}] ({s})\n", .{
        det.output_shape[0], det.output_shape[1], det.output_shape[2], @tagName(det.output_desc.format),
    });
    try w.print("Iterations          : {d} warmup + {d} measured ({d} fixed image(s))\n", .{
        cfg.warmup_iters, cfg.measured_iters, cfg.image_paths.len,
    });
    try w.print("---------------------------------------------------------------------\n", .{});
    try w.print("Startup & Resources:\n", .{});
    try w.print("  App Startup Time  : {d:.3} ms\n", .{r.app_startup_ms});
    try w.print("  Model Load Time   : {d:.3} ms\n", .{r.model_load_ms});
    try w.print("  Binary Size       : {d} bytes ({d:.2} KiB)\n", .{
        r.binary_size_bytes, @as(f64, @floatFromInt(r.binary_size_bytes)) / 1024.0,
    });
    try w.print("  Peak RSS (VmHWM)  : {d} KiB ({d:.2} MiB)\n", .{
        r.peak_rss_kb, @as(f64, @floatFromInt(r.peak_rss_kb)) / 1024.0,
    });
    try w.print("  RSS Before/After  : {d} KiB -> {d} KiB (delta={d} KiB)\n", .{
        r.rss_before_kb,
        r.rss_after_kb,
        @as(i64, @intCast(r.rss_after_kb)) - @as(i64, @intCast(r.rss_before_kb)),
    });
    try w.print("  CPU Utilization   : {d:.2}% (pipeline), {d:.2}% (inference-only) [process CPU/wall]\n", .{
        r.pipeline_cpu_util_pct, r.infer_only_cpu_util_pct,
    });
    try w.print("---------------------------------------------------------------------\n", .{});
    try w.print("Stage Latencies (ms)   |    Mean |  Median |     P95 |     P99 |     Min |     Max |     FPS\n", .{});
    try printStatsRow(w, "Preprocessing", r.preprocessing);
    try printStatsRow(w, "Inference (Pipeline)", r.pipeline_inference);
    try printStatsRow(w, "Postprocessing", r.postprocessing);
    try printStatsRow(w, "Total End-to-End", r.total_e2e);
    try printStatsRow(w, "Inference-Only (Isol)", r.inference_only);
    try w.print("=====================================================================\n\n", .{});
    try w.flush();
}

fn printStatsRow(w: *Io.Writer, label: []const u8, s: metrics.LatencyStats) !void {
    try w.print("{s:<22} | {d:7.3} | {d:7.3} | {d:7.3} | {d:7.3} | {d:7.3} | {d:7.3} | {d:7.2}\n", .{
        label,
        s.mean_ms,
        s.median_ms,
        s.p95_ms,
        s.p99_ms,
        s.min_ms,
        s.max_ms,
        s.avg_fps,
    });
}

fn writeCsvReport(
    io: Io,
    allocator: std.mem.Allocator,
    det: *const detector_mod.Detector,
    cfg: BenchmarkConfig,
    r: BenchmarkReport,
) !void {
    var aw: Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    const w = &aw.writer;

    try w.writeAll("language,zig_version,optimize_mode,ort_version,threads,warmup_iters,measured_iters,stage,count,mean_ms,median_ms,p95_ms,p99_ms,min_ms,max_ms,std_dev_ms,avg_fps,model_load_ms,app_startup_ms,peak_rss_kb,cpu_util_pct,binary_size_bytes\n");

    const rows = [_]struct { name: []const u8, stats: metrics.LatencyStats, cpu: f64 }{
        .{ .name = "preprocessing", .stats = r.preprocessing, .cpu = r.pipeline_cpu_util_pct },
        .{ .name = "pipeline_inference", .stats = r.pipeline_inference, .cpu = r.pipeline_cpu_util_pct },
        .{ .name = "postprocessing", .stats = r.postprocessing, .cpu = r.pipeline_cpu_util_pct },
        .{ .name = "total_e2e", .stats = r.total_e2e, .cpu = r.pipeline_cpu_util_pct },
        .{ .name = "inference_only", .stats = r.inference_only, .cpu = r.infer_only_cpu_util_pct },
    };

    for (rows) |row| {
        try w.print(
            "zig,{s},{s},{s},{d},{d},{d},{s},{d},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.4},{d:.6},{d:.6},{d},{d:.4},{d}\n",
            .{
                builtin.zig_version_string,
                @tagName(builtin.mode),
                det.ort_version,
                cfg.threads,
                cfg.warmup_iters,
                cfg.measured_iters,
                row.name,
                row.stats.count,
                row.stats.mean_ms,
                row.stats.median_ms,
                row.stats.p95_ms,
                row.stats.p99_ms,
                row.stats.min_ms,
                row.stats.max_ms,
                row.stats.std_dev_ms,
                row.stats.avg_fps,
                r.model_load_ms,
                r.app_startup_ms,
                r.peak_rss_kb,
                row.cpu,
                r.binary_size_bytes,
            },
        );
    }

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = cfg.csv_out_path, .data = aw.written() });
}

fn writeJsonReport(
    io: Io,
    allocator: std.mem.Allocator,
    det: *const detector_mod.Detector,
    cfg: BenchmarkConfig,
    r: BenchmarkReport,
) !void {
    var aw: Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    const w = &aw.writer;

    try w.print(
        \\{{
        \\  "language": "zig",
        \\  "zig_version": "{s}",
        \\  "optimize_mode": "{s}",
        \\  "onnxruntime_version": "{s}",
        \\  "execution_provider": "CPUExecutionProvider",
        \\  "model_path": "{s}",
        \\  "input_shape": [{d}, {d}, {d}, {d}],
        \\  "output_shape": [{d}, {d}, {d}],
        \\  "output_format": "{s}",
        \\  "config": {{
        \\    "threads": {d},
        \\    "warmup_iterations": {d},
        \\    "measured_iterations": {d},
        \\    "conf_threshold": {d:.4},
        \\    "iou_threshold": {d:.4},
        \\    "image_count": {d}
        \\  }},
        \\  "system_metrics": {{
        \\    "app_startup_ms": {d:.6},
        \\    "model_load_ms": {d:.6},
        \\    "binary_size_bytes": {d},
        \\    "peak_rss_kb": {d},
        \\    "rss_before_kb": {d},
        \\    "rss_after_kb": {d},
        \\    "pipeline_cpu_utilization_pct": {d:.4},
        \\    "inference_only_cpu_utilization_pct": {d:.4}
        \\  }},
        \\  "stages": {{
        \\
    , .{
        builtin.zig_version_string,
        @tagName(builtin.mode),
        det.ort_version,
        cfg.model_path,
        det.input_shape[0],
        det.input_shape[1],
        det.input_shape[2],
        det.input_shape[3],
        det.output_shape[0],
        det.output_shape[1],
        det.output_shape[2],
        @tagName(det.output_desc.format),
        cfg.threads,
        cfg.warmup_iters,
        cfg.measured_iters,
        cfg.conf_threshold,
        cfg.iou_threshold,
        cfg.image_paths.len,
        r.app_startup_ms,
        r.model_load_ms,
        r.binary_size_bytes,
        r.peak_rss_kb,
        r.rss_before_kb,
        r.rss_after_kb,
        r.pipeline_cpu_util_pct,
        r.infer_only_cpu_util_pct,
    });

    try writeJsonStats(w, "preprocessing", r.preprocessing, true);
    try writeJsonStats(w, "pipeline_inference", r.pipeline_inference, true);
    try writeJsonStats(w, "postprocessing", r.postprocessing, true);
    try writeJsonStats(w, "total_end_to_end", r.total_e2e, true);
    try writeJsonStats(w, "inference_only", r.inference_only, false);

    try w.writeAll("  }\n}\n");

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = cfg.json_out_path, .data = aw.written() });
}

fn writeJsonStats(w: *Io.Writer, name: []const u8, s: metrics.LatencyStats, trailing_comma: bool) !void {
    try w.print(
        \\    "{s}": {{
        \\      "count": {d},
        \\      "mean_ms": {d:.6},
        \\      "median_ms": {d:.6},
        \\      "p95_ms": {d:.6},
        \\      "p99_ms": {d:.6},
        \\      "min_ms": {d:.6},
        \\      "max_ms": {d:.6},
        \\      "std_dev_ms": {d:.6},
        \\      "avg_fps": {d:.4}
        \\    }}{s}
        \\
    , .{
        name,
        s.count,
        s.mean_ms,
        s.median_ms,
        s.p95_ms,
        s.p99_ms,
        s.min_ms,
        s.max_ms,
        s.std_dev_ms,
        s.avg_fps,
        if (trailing_comma) "," else "",
    });
}
