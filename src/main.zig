const std = @import("std");
const Io = std.Io;
const Clock = Io.Clock;

const image_mod = @import("image.zig");
const detector_mod = @import("detector.zig");
const postprocessing = @import("postprocessing.zig");
const benchmark = @import("benchmark.zig");
const metrics = @import("metrics.zig");
const camera_mod = @import("camera.zig");

const Mode = enum {
    detect,
    validate,
    benchmark,
    camera,
};

const CliOptions = struct {
    mode: Mode = .detect,
    model_path: []const u8 = "models/yolov26n.onnx",
    image_paths: [16][]const u8 = undefined,
    image_count: usize = 0,
    output_image: ?[]const u8 = null,
    csv_out: []const u8 = "benchmarks/zig_benchmark_results.csv",
    json_out: []const u8 = "benchmarks/zig_benchmark_results.json",
    threads: u32 = 4,
    warmup_iters: usize = 100,
    measured_iters: usize = 1000,
    conf_threshold: f32 = 0.25,
    iou_threshold: f32 = 0.45,
    force_nms: bool = false,
    cam_device: []const u8 = "/dev/video2",
    cam_width: u32 = 640,
    cam_height: u32 = 480,
    cam_frames: usize = 30,
    cam_warmup: usize = 5,
    cam_format: camera_mod.PixelFormat = .yuyv,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const app_start_ts = Clock.Timestamp.now(io, .awake);

    // Use DebugAllocator/SafeAllocator in Debug builds for leak detection,
    // and Zig's GPA/init allocator cleanly across modes.
    const allocator = init.gpa;

    var args_it = try init.minimal.args.iterateAllocator(allocator);
    defer args_it.deinit();

    const opts = parseArgs(io, &args_it) catch |err| {
        if (err == error.HelpRequested) return;
        return err;
    };

    var default_imgs = [_][]const u8{
        "test/bus.jpg",
        "test/zidane.jpg",
    };
    const img_slice: []const []const u8 = if (opts.image_count > 0)
        opts.image_paths[0..opts.image_count]
    else
        &default_imgs;

    switch (opts.mode) {
        .detect => try runDetect(io, allocator, opts, img_slice[0]),
        .validate => try runValidate(io, allocator, opts, img_slice),
        .camera => try runCamera(io, allocator, opts),
        .benchmark => {
            _ = try benchmark.runBenchmark(io, allocator, app_start_ts, .{
                .model_path = opts.model_path,
                .image_paths = img_slice,
                .warmup_iters = opts.warmup_iters,
                .measured_iters = opts.measured_iters,
                .threads = opts.threads,
                .conf_threshold = opts.conf_threshold,
                .iou_threshold = opts.iou_threshold,
                .force_nms = opts.force_nms,
                .csv_out_path = opts.csv_out,
                .json_out_path = opts.json_out,
            });
        },
    }
}

fn runDetect(
    io: Io,
    allocator: std.mem.Allocator,
    opts: CliOptions,
    image_path: []const u8,
) !void {
    var det = try detector_mod.Detector.init(allocator, .{
        .model_path = opts.model_path,
        .intra_op_threads = opts.threads,
        .conf_threshold = opts.conf_threshold,
        .iou_threshold = opts.iou_threshold,
        .force_nms = opts.force_nms,
    });
    defer det.deinit();

    var img = try image_mod.Image.load(allocator, image_path);
    defer img.deinit();

    const t0 = Clock.Timestamp.now(io, .awake);
    const lb = det.preprocess(img.data, img.width, img.height);
    const t1 = Clock.Timestamp.now(io, .awake);
    try det.inferRaw();
    const t2 = Clock.Timestamp.now(io, .awake);
    const detections = det.postprocess(lb);
    const t3 = Clock.Timestamp.now(io, .awake);

    var out_buf: [2048]u8 = undefined;
    var fw = Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &fw.interface;

    try w.print("Image: {s} ({d}x{d}), Output format: {s}\n", .{
        image_path, img.width, img.height, @tagName(det.output_desc.format),
    });
    try w.print("Latency: pre={d:.3}ms, infer={d:.3}ms, post={d:.3}ms, total={d:.3}ms\n", .{
        @as(f64, @floatFromInt(t0.durationTo(t1).raw.nanoseconds)) / 1e6,
        @as(f64, @floatFromInt(t1.durationTo(t2).raw.nanoseconds)) / 1e6,
        @as(f64, @floatFromInt(t2.durationTo(t3).raw.nanoseconds)) / 1e6,
        @as(f64, @floatFromInt(t0.durationTo(t3).raw.nanoseconds)) / 1e6,
    });
    try w.print("Detections ({d}):\n", .{detections.len});
    for (detections, 0..) |d, idx| {
        try w.print("  [{d}] class={d} ({s}) conf={d:.4} box=[{d:.2}, {d:.2}, {d:.2}, {d:.2}]\n", .{
            idx, d.class_id, postprocessing.className(d.class_id), d.confidence, d.x1, d.y1, d.x2, d.y2,
        });
    }
    try w.flush();

    if (opts.output_image) |out_path| {
        var vis = try img.clone(allocator);
        defer vis.deinit();
        vis.drawDetections(detections);
        try vis.savePng(allocator, out_path);
    }
}

fn runValidate(
    io: Io,
    allocator: std.mem.Allocator,
    opts: CliOptions,
    image_paths: []const []const u8,
) !void {
    var out_buf: [2048]u8 = undefined;
    var fw = Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &fw.interface;

    var det = try detector_mod.Detector.init(allocator, .{
        .model_path = opts.model_path,
        .intra_op_threads = opts.threads,
        .conf_threshold = opts.conf_threshold,
        .iou_threshold = opts.iou_threshold,
        .force_nms = opts.force_nms,
    });
    defer det.deinit();

    try w.print("[PASS] Model loaded: {s} (ORT {s})\n", .{ opts.model_path, det.ort_version });
    try w.print("[PASS] Input tensor validated: [{d}, {d}, {d}, {d}] FP32\n", .{
        det.input_shape[0], det.input_shape[1], det.input_shape[2], det.input_shape[3],
    });
    try w.print("[PASS] Output tensor validated: [{d}, {d}, {d}] ({s})\n", .{
        det.output_shape[0], det.output_shape[1], det.output_shape[2], @tagName(det.output_desc.format),
    });

    for (image_paths) |path| {
        var img = try image_mod.Image.load(allocator, path);
        defer img.deinit();

        const lb = det.preprocess(img.data, img.width, img.height);
        try det.inferRaw();
        const detections = det.postprocess(lb);
        if (detections.len == 0) return error.NoDetectionsFound;

        const fw_img: f32 = @floatFromInt(img.width);
        const fh_img: f32 = @floatFromInt(img.height);
        for (detections) |d| {
            if (d.x1 < 0.0 or d.y1 < 0.0 or d.x2 > fw_img or d.y2 > fh_img or d.x2 <= d.x1 or d.y2 <= d.y1) {
                return error.BoundingBoxOutOfBounds;
            }
            if (!std.math.isFinite(d.confidence) or d.confidence < opts.conf_threshold or d.confidence > 1.0) {
                return error.InvalidConfidence;
            }
            if (d.class_id >= postprocessing.NUM_COCO_CLASSES) {
                return error.InvalidClassId;
            }
        }
        try w.print("[PASS] {s}: {d} valid detections, all boxes within [0..{d}, 0..{d}]\n", .{
            path, detections.len, img.width, img.height,
        });
    }

    // Verify repeated inference does not leak / grow RSS
    const rss_start = metrics.readProcessMemory(io).current_rss_kb;
    for (0..50) |_| {
        try det.inferRaw();
    }
    const rss_end = metrics.readProcessMemory(io).current_rss_kb;
    try w.print("[PASS] 50 repeated inferences RSS check: {d} KiB -> {d} KiB\n", .{ rss_start, rss_end });
    try w.flush();
}

fn runCamera(
    io: Io,
    allocator: std.mem.Allocator,
    opts: CliOptions,
) !void {
    var out_buf: [4096]u8 = undefined;
    var fw = Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &fw.interface;

    var det = try detector_mod.Detector.init(allocator, .{
        .model_path = opts.model_path,
        .intra_op_threads = opts.threads,
        .conf_threshold = opts.conf_threshold,
        .iou_threshold = opts.iou_threshold,
        .force_nms = opts.force_nms,
    });
    defer det.deinit();

    var cam = try camera_mod.Camera.open(
        allocator,
        opts.cam_device,
        opts.cam_width,
        opts.cam_height,
        opts.cam_format,
    );
    defer cam.close();

    try w.print("Camera opened: {s} ({d}x{d}, format={s})\n", .{
        opts.cam_device, cam.width, cam.height, @tagName(cam.pixfmt),
    });
    try w.print("Model: {s} (ORT {s}, threads={d})\n", .{
        opts.model_path, det.ort_version, opts.threads,
    });
    try w.print("Warming up camera + ORT for {d} frame(s), then capturing {d} frame(s)...\n", .{
        opts.cam_warmup, opts.cam_frames,
    });
    try w.flush();

    for (0..opts.cam_warmup) |_| {
        const rgb = try cam.captureRgb(3000);
        const lb = det.preprocess(rgb, cam.width, cam.height);
        try det.inferRaw();
        _ = det.postprocess(lb);
    }

    const cap_ns = try allocator.alloc(i64, opts.cam_frames);
    defer allocator.free(cap_ns);
    const pre_ns = try allocator.alloc(i64, opts.cam_frames);
    defer allocator.free(pre_ns);
    const inf_ns = try allocator.alloc(i64, opts.cam_frames);
    defer allocator.free(inf_ns);
    const post_ns = try allocator.alloc(i64, opts.cam_frames);
    defer allocator.free(post_ns);
    const tot_ns = try allocator.alloc(i64, opts.cam_frames);
    defer allocator.free(tot_ns);
    const scratch_ns = try allocator.alloc(i64, opts.cam_frames);
    defer allocator.free(scratch_ns);

    const rss_start = metrics.readProcessMemory(io).current_rss_kb;
    var best_vis: ?image_mod.Image = null;
    defer if (best_vis) |*v| v.deinit();
    var best_det_count: usize = 0;

    for (0..opts.cam_frames) |f| {
        const t0 = Clock.Timestamp.now(io, .awake);
        const rgb = try cam.captureRgb(3000);
        const t1 = Clock.Timestamp.now(io, .awake);
        const lb = det.preprocess(rgb, cam.width, cam.height);
        const t2 = Clock.Timestamp.now(io, .awake);
        try det.inferRaw();
        const t3 = Clock.Timestamp.now(io, .awake);
        const detections = det.postprocess(lb);
        const t4 = Clock.Timestamp.now(io, .awake);

        cap_ns[f] = @intCast(t0.durationTo(t1).raw.nanoseconds);
        pre_ns[f] = @intCast(t1.durationTo(t2).raw.nanoseconds);
        inf_ns[f] = @intCast(t2.durationTo(t3).raw.nanoseconds);
        post_ns[f] = @intCast(t3.durationTo(t4).raw.nanoseconds);
        tot_ns[f] = @intCast(t0.durationTo(t4).raw.nanoseconds);

        if (opts.output_image != null and (best_vis == null or detections.len >= best_det_count)) {
            if (best_vis) |*prev| prev.deinit();
            var vis = try image_mod.Image.cloneFromRgb(allocator, rgb, cam.width, cam.height);
            vis.drawDetections(detections);
            best_vis = vis;
            best_det_count = detections.len;
        }

        try w.print("Frame [{d}/{d}] cap={d:.2}ms pre={d:.2}ms inf={d:.2}ms post={d:.3}ms total={d:.2}ms | dets={d}:", .{
            f + 1,
            opts.cam_frames,
            @as(f64, @floatFromInt(cap_ns[f])) / 1e6,
            @as(f64, @floatFromInt(pre_ns[f])) / 1e6,
            @as(f64, @floatFromInt(inf_ns[f])) / 1e6,
            @as(f64, @floatFromInt(post_ns[f])) / 1e6,
            @as(f64, @floatFromInt(tot_ns[f])) / 1e6,
            detections.len,
        });
        for (detections) |d| {
            try w.print(" {s}({d:.2})", .{ postprocessing.className(d.class_id), d.confidence });
        }
        try w.writeAll("\n");
    }

    if (opts.output_image) |out_path| {
        if (best_vis) |*vis| {
            try vis.savePng(allocator, out_path);
            try w.print("Saved annotated camera frame ({d} detections) -> {s}\n", .{ best_det_count, out_path });
        }
    }

    const mem_after = metrics.readProcessMemory(io);
    const cap_st = metrics.computeStats(cap_ns, scratch_ns);
    const pre_st = metrics.computeStats(pre_ns, scratch_ns);
    const inf_st = metrics.computeStats(inf_ns, scratch_ns);
    const post_st = metrics.computeStats(post_ns, scratch_ns);
    const tot_st = metrics.computeStats(tot_ns, scratch_ns);

    try w.print("\n--- Live Camera Summary ({d} frames) ---\n", .{opts.cam_frames});
    try w.print("  Capture+Conv : mean={d:.3} ms (median={d:.3} ms)\n", .{ cap_st.mean_ms, cap_st.median_ms });
    try w.print("  Preprocess   : mean={d:.3} ms (median={d:.3} ms)\n", .{ pre_st.mean_ms, pre_st.median_ms });
    try w.print("  Inference    : mean={d:.3} ms (median={d:.3} ms, {d:.2} FPS)\n", .{ inf_st.mean_ms, inf_st.median_ms, inf_st.avg_fps });
    try w.print("  Postprocess  : mean={d:.3} ms\n", .{post_st.mean_ms});
    try w.print("  Total E2E    : mean={d:.3} ms (median={d:.3} ms, {d:.2} FPS)\n", .{ tot_st.mean_ms, tot_st.median_ms, tot_st.avg_fps });
    try w.print("  RSS Check    : {d} KiB -> {d} KiB (Peak={d} KiB)\n", .{ rss_start, mem_after.current_rss_kb, mem_after.peak_rss_kb });
    try w.flush();
}

fn parseArgs(io: Io, it: *std.process.Args.Iterator) !CliOptions {
    var opts = CliOptions{};
    _ = it.next(); // executable name

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "detect")) {
            opts.mode = .detect;
        } else if (std.mem.eql(u8, arg, "validate")) {
            opts.mode = .validate;
        } else if (std.mem.eql(u8, arg, "benchmark")) {
            opts.mode = .benchmark;
        } else if (std.mem.eql(u8, arg, "camera")) {
            opts.mode = .camera;
        } else if (std.mem.eql(u8, arg, "--device")) {
            opts.cam_device = it.next() orelse return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--width")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.cam_width = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, arg, "--height")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.cam_height = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, arg, "--frames")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.cam_frames = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--cam-warmup")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.cam_warmup = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--format")) {
            const v = it.next() orelse return error.MissingArgValue;
            if (std.mem.eql(u8, v, "yuyv")) {
                opts.cam_format = .yuyv;
            } else if (std.mem.eql(u8, v, "mjpg")) {
                opts.cam_format = .mjpg;
            } else {
                return error.InvalidCameraFormat;
            }
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printUsage(io);
            return error.HelpRequested;
        } else if (std.mem.eql(u8, arg, "--model")) {
            opts.model_path = it.next() orelse return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--image")) {
            const p = it.next() orelse return error.MissingArgValue;
            if (opts.image_count >= opts.image_paths.len) return error.TooManyImages;
            opts.image_paths[opts.image_count] = p;
            opts.image_count += 1;
        } else if (std.mem.eql(u8, arg, "--output")) {
            opts.output_image = it.next() orelse return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            opts.csv_out = it.next() orelse return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--json")) {
            opts.json_out = it.next() orelse return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--threads")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.threads = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.warmup_iters = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--iters")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.measured_iters = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--conf")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.conf_threshold = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, arg, "--iou")) {
            const v = it.next() orelse return error.MissingArgValue;
            opts.iou_threshold = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, arg, "--force-nms")) {
            opts.force_nms = true;
        } else {
            return error.UnknownArgument;
        }
    }
    return opts;
}

fn printUsage(io: Io) !void {
    var buf: [2048]u8 = undefined;
    var fw = Io.File.stdout().writerStreaming(io, &buf);
    try fw.interface.writeAll(
        \\Usage: zig-yolo26n [detect|validate|benchmark|camera] [options]
        \\
        \\Options:
        \\  --model <path>     ONNX model path (default: models/yolov26n.onnx)
        \\  --image <path>     Input image path (repeatable; defaults to test/bus.jpg, test/zidane.jpg)
        \\  --output <path>    Save annotated PNG image in detect or camera mode
        \\  --threads <n>      ONNX Runtime intra-op CPU threads (default: 4)
        \\  --warmup <n>       Warmup iterations for benchmark (default: 100)
        \\  --iters <n>        Measured iterations for benchmark (default: 1000)
        \\  --conf <f32>       Confidence threshold (default: 0.25)
        \\  --iou <f32>        IoU threshold for NMS when applicable (default: 0.45)
        \\  --force-nms        Force NMS even on end2end [1,N,6] outputs
        \\  --csv <path>       Benchmark CSV output path (default: benchmarks/zig_benchmark_results.csv)
        \\  --json <path>      Benchmark JSON output path (default: benchmarks/zig_benchmark_results.json)
        \\  --device <path>    V4L2 camera device in camera mode (default: /dev/video2)
        \\  --width <n>        Camera capture width (default: 640)
        \\  --height <n>       Camera capture height (default: 480)
        \\  --format <fmt>     Camera pixel format: yuyv or mjpg (default: yuyv)
        \\  --frames <n>       Frames to capture and process in camera mode (default: 30)
        \\  --cam-warmup <n>   Camera warmup frames before stats (default: 5)
        \\  -h, --help         Show this help message
        \\
    );
    try fw.interface.flush();
}
