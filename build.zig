const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ort_include = b.option(
        []const u8,
        "ort-include",
        "Path to directory containing onnxruntime_c_api.h (defaults to local include/)",
    );
    const ort_lib = b.option(
        []const u8,
        "ort-lib",
        "Path to directory containing libonnxruntime.so (defaults to local lib/)",
    );

    const ort_translate = b.addTranslateC(.{
        .root_source_file = b.path("include/onnxruntime_c_api.h"),
        .target = target,
        .optimize = optimize,
    });
    ort_translate.addIncludePath(b.path("include"));
    if (ort_include) |inc_path| {
        ort_translate.addIncludePath(.{ .cwd_relative = inc_path });
    }

    const stb_translate = b.addTranslateC(.{
        .root_source_file = b.path("include/stb_c.h"),
        .target = target,
        .optimize = optimize,
    });
    stb_translate.addIncludePath(b.path("include"));

    const cam_translate = b.addTranslateC(.{
        .root_source_file = b.path("include/camera_v4l2.h"),
        .target = target,
        .optimize = optimize,
    });
    cam_translate.addIncludePath(b.path("include"));

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    root_module.addImport("ort_c", ort_translate.createModule());
    root_module.addImport("stb_c", stb_translate.createModule());
    root_module.addImport("camera_c", cam_translate.createModule());
    root_module.addIncludePath(b.path("include"));
    if (ort_include) |inc_path| {
        root_module.addIncludePath(.{ .cwd_relative = inc_path });
    }

    root_module.addLibraryPath(b.path("lib"));
    if (ort_lib) |lib_path| {
        root_module.addLibraryPath(.{ .cwd_relative = lib_path });
        root_module.addRPath(.{ .cwd_relative = lib_path });
    }
    // Embed RPATH relative to zig-out/bin -> ../../lib and system /usr/lib fallback
    root_module.addRPathSpecial("$ORIGIN/../../lib");
    root_module.addRPathSpecial("$ORIGIN/../lib");
    root_module.addRPathSpecial("/usr/lib");

    root_module.addCSourceFile(.{
        .file = b.path("src/stb_impl.c"),
        .flags = &.{"-O3"},
    });
    root_module.addCSourceFile(.{
        .file = b.path("src/camera_v4l2.c"),
        .flags = &.{"-O3"},
    });

    root_module.linkSystemLibrary("onnxruntime", .{});
    root_module.linkSystemLibrary("m", .{});

    const exe = b.addExecutable(.{
        .name = "zig-yolo26n",
        .root_module = root_module,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the zig-yolo26n application");
    run_step.dependOn(&run_cmd.step);

    const bench_cmd = b.addRunArtifact(exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    bench_cmd.addArg("benchmark");
    const bench_step = b.step("benchmark", "Run the YOLO26n benchmark");
    bench_step.dependOn(&bench_cmd.step);

    // Pure-Zig unit tests for preprocessing, postprocessing, and metrics
    const test_preproc_mod = b.createModule(.{
        .root_source_file = b.path("src/preprocessing.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_postproc_mod = b.createModule(.{
        .root_source_file = b.path("src/postprocessing.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_metrics_mod = b.createModule(.{
        .root_source_file = b.path("src/metrics.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const test_cam_mod = b.createModule(.{
        .root_source_file = b.path("src/camera.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_cam_mod.addImport("stb_c", stb_translate.createModule());
    test_cam_mod.addImport("camera_c", cam_translate.createModule());
    test_cam_mod.addCSourceFile(.{ .file = b.path("src/stb_impl.c"), .flags = &.{"-O3"} });
    test_cam_mod.addCSourceFile(.{ .file = b.path("src/camera_v4l2.c"), .flags = &.{"-O3"} });
    test_cam_mod.addIncludePath(b.path("include"));
    test_cam_mod.linkSystemLibrary("m", .{});

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = test_preproc_mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = test_postproc_mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = test_metrics_mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = test_cam_mod })).step);
}
