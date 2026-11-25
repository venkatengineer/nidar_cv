# Zig Edge Computer Vision Benchmark — YOLO26n (ONNX Runtime CPU)

A modular, memory-conscious edge computer vision inference and benchmarking application written in **Zig 0.17.1**, running **YOLO26n (FP32, 640×640)** through the **ONNX Runtime 1.29.0 C API** on Linux `x86_64` CPU.

This project establishes a controlled, reproducible Zig baseline for cross-language benchmarking against future C++ and Rust implementations using identical ONNX weights, ONNX Runtime version, preprocessing, input images, and measurement methodology.

---

## 1. Directory Structure

```
zig-human-detection/
├── src/
│   ├── main.zig            # CLI entry point (detect, validate, benchmark, camera)
│   ├── detector.zig        # ONNX Runtime C API session & zero-copy tensor lifecycle
│   ├── image.zig           # stb_image / stb_image_write wrapper & box visualization
│   ├── camera.zig          # Native Linux V4L2 camera capture & YUYV/MJPG decode
│   ├── camera_v4l2.c       # Minimal Linux V4L2 mmap streaming C helper
│   ├── preprocessing.zig   # Deterministic letterbox + bilinear RGB -> NCHW FP32
│   ├── postprocessing.zig  # YOLO26 [1, 300, 6] & raw [1, 84, 8400] decoders + NMS
│   ├── benchmark.zig       # 100-warmup + 1,000-iteration runner, CSV/JSON export
│   ├── metrics.zig         # Monotonic clock stats, /proc/self/status RSS, CPU util
│   └── stb_impl.c          # Compilation unit for stb_image & stb_image_write
├── models/
│   └── yolov26n.onnx       # Official Ultralytics YOLO26n exported to FP32 ONNX
├── include/                # ONNX Runtime C API headers + stb_image headers
├── lib/                    # libonnxruntime.so symlinks / shared libraries
├── benchmarks/             # Exported CSV, JSON, and visualization PNG outputs
├── test/                   # Fixed reproducible test images (bus.jpg, zidane.jpg)
├── build.zig               # Zig 0.17.1 build configuration (Debug, ReleaseFast, test)
├── README.md
└── .gitignore
```

---

## 2. Model Setup & Provenance

- **Model**: Official Ultralytics `yolo26n.pt` (COCO 80 classes, FP32, `640×640`)
- **Export tool**: `ultralytics==8.4.218`, `torch==2.15.0+cpu`, `onnx==1.23.2`, `onnxslim==0.1.99`
- **Source weights**: `yolo26n.pt` (`5,512,920` bytes / `5.3M`, SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`)
- **Exported ONNX (`models/yolov26n.onnx`)**:
  - **File size**: `9,942,097` bytes (`9.48 MiB`)
  - **SHA-256**: `8ef199e41e23b930853d0e34a5ae5d5c291c0b9abf3898303e61d9363db2b4d3`
  - **ONNX Opset**: `18`
  - **Input tensor**: `images` — `FP32`, shape `[1, 3, 640, 640]` (NCHW, RGB, `[0.0, 1.0]`)
  - **Output tensor**: `output0` — `FP32`, shape `[1, 300, 6]` (`end2end: True`)

### Reproducing the ONNX Export (Optional External Python Step)

Python is **never** used at runtime. To regenerate `models/yolov26n.onnx` externally:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
# nms=False activates YOLO26's native end-to-end one-to-one NMS-free head -> [1, 300, 6]
model.export(format="onnx", imgsz=640, half=False, dynamic=False, simplify=True, nms=False)
```

### YOLO26 vs YOLOv8 Output Format

Unlike YOLOv8 (`[1, 84, 8400]` raw anchors requiring class-wise NMS), YOLO26's native one-to-one head (`end2end=True`) outputs `[1, 300, 6]`, where each row is `[x1, y1, x2, y2, confidence, class_id]` in letterboxed `640×640` pixel coordinates and does **not** require NMS.

`src/postprocessing.zig` inspects the ONNX output tensor dimensions at startup (`inferOutputDescriptor`) and supports:
- `[1, N, 6]` (`yolo26_end2end`, NMS-free by default; optional `--force-nms` flag)
- `[1, 84, N]` (`yolo_raw_ch_first`, fallback with class-aware NMS)
- `[1, N, 84]` (`yolo_raw_ch_last`, fallback with class-aware NMS)

---

## 3. Fixed Preprocessing & Test Inputs

For cross-language parity, preprocessing is strictly deterministic across Zig, C++, and Rust:

1. Decode image to 3-channel RGB24 (`u8`).
2. Compute aspect-ratio-preserving scale `min(640 / orig_w, 640 / orig_h)` and Ultralytics-aligned symmetric padding (`pad = round(half_diff - 0.1)`).
3. Fill the `[3, 640, 640]` NCHW `f32` buffer with normalized padding `114.0 / 255.0`.
4. Bilinear-resize using pixel-center mapping `((dst + 0.5) * src_dim / resized_dim) - 0.5` directly into planar `R`, `G`, `B` channels divided by `255.0`.

Fixed test inputs in `test/`:
- `test/bus.jpg` (`810×1080`, `137,794` bytes, SHA-256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`)
- `test/zidane.jpg` (`1280×720`, `50,247` bytes, SHA-256: `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb`)

---

## 4. Dependencies & Build Instructions

### External Dependencies

- **Zig**: `0.17.1` (official `x86_64-linux` release)
- **ONNX Runtime (CPU)**: `1.29.0` (`libonnxruntime.so.1.29.0`, C API header `ORT_API_VERSION = 29`)
- **C Image Headers**: `stb_image.h` & `stb_image_write.h` (included in `include/`)
- **libc & libm**: Standard Linux `glibc` and math library

In Zig `0.17.1`, C headers are translated at build time via `b.addTranslateC(...)` in `build.zig` (`ort_c` and `stb_c` modules), with RPATH embedded for `lib/` and `/usr/lib`. You can also override paths with `-Dort-include=<dir> -Dort-lib=<dir>`.

### Build & Run Commands

```bash
cd zig-human-detection

# 1. Run pure-Zig unit tests (preprocessing, postprocessing, metrics)
zig build test

# 2. Build Debug binary (includes leak checks via std.process.Init gpa)
zig build -Doptimize=Debug

# 3. Build ReleaseFast binary (used for benchmarking)
zig build -Doptimize=ReleaseFast

# 4. Validate model load, tensor dimensions, detection bounds, and RSS stability
./zig-out/bin/zig-yolo26n validate

# 5. Run single-image detection with optional annotated PNG output
./zig-out/bin/zig-yolo26n detect --image test/bus.jpg --output benchmarks/bus_detected.png

# 6. Run reproducible benchmark (100 warmup + 1000 measured iterations, 4 CPU threads)
./zig-out/bin/zig-yolo26n benchmark --threads 4 --warmup 100 --iters 1000

# 7. Run live external V4L2 camera inference (e.g., Logitech C922 on /dev/video2)
./zig-out/bin/zig-yolo26n camera --device /dev/video2 --width 640 --height 480 --format yuyv --frames 30 --output benchmarks/camera_yuyv_detected.png
```

---

## 5. Memory Management & Zig Integration Decisions

- **Explicit Allocator**: Uses Zig's `init.gpa` (`std.process.Init`) passed explicitly through `Detector.init`, `Image.load`, and `BenchmarkConfig`.
- **Zero Per-Frame Heap Allocations**:
  - `input_tensor_buf` (`1 × 3 × 640 × 640 × 4 = 4,915,200` bytes) and `output_tensor_buf` (`1 × 300 × 6 × 4 = 7,200` bytes) are allocated once in `Detector.init()`.
  - `CreateTensorWithDataAsOrtValue` wraps those Zig-owned buffers into reusable `OrtValue` handles (`input_ort_value`, `output_ort_value`), so `OrtApi.Run` executes zero-copy without allocating or freeing tensors per inference.
  - `ScratchBuffer` and `detections_buf` are preallocated once for postprocessing.
- **Deterministic Cleanup**: `Detector.deinit()` releases `OrtValue`s, `OrtMemoryInfo`, `OrtSession`, `OrtSessionOptions`, `OrtEnv`, and all Zig slices in reverse order. Debug mode exits with zero reported leaks.
- **Session Configuration**: Configured strictly for CPU execution (`ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`, `EnableCpuMemArena`, `EnableMemPattern`, `inter_op_num_threads=1`, `intra_op_num_threads=4`).

---

## 6. Observed System Specifications & Benchmark Results

### Test System Specifications

- **OS**: Arch Linux (`Linux solaris 7.2.3-zen1-2-zen x86_64`)
- **CPU**: Intel(R) Core(TM) 7 240H (10 cores / 16 logical threads, max 5.20 GHz, AVX2 / AVX-VNNI)
- **RAM**: 15 GiB
- **Zig**: `0.17.1` (`-Doptimize=ReleaseFast`)
- **ONNX Runtime**: `1.29.0` (`CPUExecutionProvider`, `intra_op_num_threads=4`, `inter_op_num_threads=1`)

### Validation Results

Verified against Ultralytics Python ONNX Runtime (`1.32.0`) reference predictions:
- `test/bus.jpg`: 5 detections (`bus` 0.9262, 4× `person` 0.9161 / 0.8993 / 0.8614 / 0.5216), all boxes strictly within `[0..810, 0..1080]`.
- `test/zidane.jpg`: 3 detections (`person` 0.9199, `person` 0.9024, `tie` 0.5083), all boxes strictly within `[0..1280, 0..720]`.
- 50 repeated inferences in `validate`: `106,944 KiB -> 106,944 KiB` (`0 KiB` RSS growth).

### Benchmark Results (`100` warmup + `1,000` measured iterations, `threads=4`)

Saved in [benchmarks/zig_benchmark_results.csv](benchmarks/zig_benchmark_results.csv) and [benchmarks/zig_benchmark_results.json](benchmarks/zig_benchmark_results.json):

| Metric | Measured Value |
|---|---|
| Application Startup Time | `58.631 ms` |
| Model Loading Time | `49.930 ms` |
| Executable Binary Size (`ReleaseFast`, unstripped ELF) | `5,532,648 bytes` (`5,402.98 KiB`; text+data+bss = `1,128,739 bytes`) |
| Peak Resident Memory (`VmHWM`) | `110,152 KiB` (`107.57 MiB`) |
| RSS Before Warmup -> After Benchmark | `72,264 KiB -> 110,152 KiB` (ORT CPU arena populated during warmup) |
| CPU Utilization (process CPU / wall time) | `397.40%` (pipeline), `397.11%` (inference-only) |

| Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | Avg FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preprocessing | `3.418` | `3.479` | `4.045` | `4.660` | `2.610` | `6.103` | `292.57` |
| Inference (Pipeline) | `18.814` | `18.676` | `20.429` | `21.946` | `16.463` | `23.609` | `53.15` |
| Postprocessing | `0.001` | `0.001` | `0.001` | `0.002` | `0.001` | `0.005` | `1,013,746.40` |
| **Total End-to-End** | **`22.233`** | **`22.095`** | **`24.124`** | **`25.551`** | **`19.387`** | **`28.372`** | **`44.98`** |
| **Inference-Only (Isolated)** | **`19.732`** | **`19.348`** | **`22.027`** | **`23.404`** | **`17.358`** | **`27.659`** | **`50.68`** |

### Live External Camera Test (`Logitech C922 Pro Stream Webcam`, `/dev/video2`)

Tested with native Linux V4L2 `mmap` streaming (`ReleaseFast`, `threads=4`):
- **640×480 `YUYV` (30 frames, 5 warmup)**: Capture+YUYV→RGB mean `16.805 ms`, Preprocess `3.492 ms`, Inference `17.895 ms` (`55.88 FPS`), Postprocess `0.001 ms`, Total E2E `38.193 ms` (`26.18 FPS`), Peak RSS `108,212 KiB`.
- **1280×720 `MJPG` (15 frames, 5 warmup)**: Capture+MJPG→RGB mean `20.547 ms`, Preprocess `2.871 ms`, Inference `17.823 ms` (`56.11 FPS`), Postprocess `0.001 ms`, Total E2E `41.242 ms` (`24.25 FPS`), Peak RSS `119,616 KiB`.
- *(Note: During this test run, the camera feed was dark — pixel values `0..8`, mean `1.91..3.04 / 255` — confirmed identically with `ffmpeg`, so 0 objects were detected in the dark frame.)*

---

## 7. Known Limitations & Next Steps

- **Preloaded Images**: The benchmark preloads the decoded RGB images into memory before the 100+1,000 loop to isolate compute latency from disk I/O and JPEG decoding overhead. If JPEG decoding throughput is also to be compared later, a separate decode-inclusive benchmark mode can be added.
- **Unstripped ReleaseFast Binary**: `zig build -Doptimize=ReleaseFast` retains DWARF debug info (`5.53 MB` file size vs `1.13 MB` loaded ELF sections). Stripping symbols (`-Dstrip` / `strip -s`) reduces disk footprint for edge deployment.
- **Fixed Image Set Size**: Current benchmarks cycle through 2 standard COCO images (`bus.jpg`, `zidane.jpg`). Expanding `test/` with a larger fixed subset of COCO `val2017` images requires no code changes (pass multiple `--image` flags).
- **Next Steps**: Implement identical C++ and Rust benchmark harnesses targeting the exact same `models/yolov26n.onnx`, `libonnxruntime.so.1.29.0`, preprocessing pipeline, and test images, then aggregate CSV/JSON comparisons.
