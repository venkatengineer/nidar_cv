# C++17 Edge Computer Vision Benchmark — YOLO26n (ONNX Runtime CPU)

Controlled **C++17** equivalent of the [Zig 0.17.1 YOLO26n benchmark](../zig-human-detection/README.md), running **YOLO26n (FP32, 640×640)** through the **ONNX Runtime 1.29.0 C API** on Linux `x86_64` CPU for direct cross-language performance comparison.

---

## 1. Directory Structure

```
cpp-human-detection/
├── src/
│   ├── main.cpp            # CLI entry point (detect, validate, benchmark)
│   ├── detector.hpp/.cpp   # ONNX Runtime C API session & zero-copy tensor lifecycle
│   ├── image.hpp/.cpp      # stb_image / stb_image_write wrapper & box visualization
│   ├── preprocessing.hpp/.cpp   # Deterministic letterbox + bilinear RGB -> NCHW FP32
│   ├── postprocessing.hpp/.cpp  # YOLO26 [1, 300, 6] & raw [1, 84, 8400] decoders + NMS
│   ├── benchmark.hpp/.cpp  # 100-warmup + 1,000-iteration runner, CSV/JSON export
│   ├── metrics.hpp/.cpp    # CLOCK_MONOTONIC stats, /proc/self/status RSS, CPU util
│   ├── stb_impl.c          # Compilation unit for stb_image & stb_image_write
│   └── unit_tests.cpp      # CTest unit test suite
├── models/
│   └── yolov26n.onnx       # Identical ONNX model (SHA-256: 8ef199e41e23b930...)
├── include/                # ONNX Runtime 1.29.0 C API + stb_image headers
├── lib/                    # libonnxruntime.so symlinks
├── test/                   # Identical fixed test images (bus.jpg, zidane.jpg)
├── benchmarks/             # Trial 1/2/3 CSV & JSON results + annotated PNGs
├── CMakeLists.txt
└── README.md
```

---

## 2. Fairness & Controlled Conditions

All parameters are held strictly identical to the Zig baseline:
- **Hardware / OS**: Intel Core 7 240H (`x86_64`), Arch Linux (`Linux solaris 7.2.3-zen1-2-zen`)
- **Model**: `models/yolov26n.onnx` (`9,942,097` bytes, SHA-256: `8ef199e41e23b930853d0e34a5ae5d5c291c0b9abf3898303e61d9363db2b4d3`, output `[1, 300, 6]`)
- **Runtime**: ONNX Runtime `1.29.0` C API (`ORT_API_VERSION = 29`), `CPUExecutionProvider`, `intra_op_threads = 4`, `inter_op_threads = 1`, `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`, `EnableCpuMemArena`, `EnableMemPattern`
- **Image Loading & Preprocessing**: Identical `stb_image.h` (`RGB24`), aspect-preserving letterbox (`114/255` pad), OpenCV pixel-center bilinear resize, `[0, 1]` FP32 NCHW normalization, zero per-frame allocations
- **Postprocessing**: Identical `[1, 300, 6]` YOLO26 end-to-end decoder (`conf=0.25`, `iou=0.45`, 80 COCO classes, zero per-frame allocations)
- **Test Inputs**: `test/bus.jpg` (`c02019c4...`) and `test/zidane.jpg` (`16d73869...`)
- **Compiler & Build Configuration**:
  - `g++ (GCC) 16.2.1 20260810`, `-std=c++17`, `CMake 4.4.2`
  - **Release**: `-DCMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`)
  - **Debug**: `-DCMAKE_BUILD_TYPE=Debug` (`-g`)

---

## 3. Build, Test, and Execution Instructions

```bash
cd cpp-human-detection

# 1. Debug build + unit tests + validation
cmake -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ctest --test-dir build-debug --output-on-failure
./build-debug/cpp-yolo26n validate

# 2. Release build (-O3 -DNDEBUG) + unit tests + validation
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/cpp-yolo26n validate

# 3. Single-image detection + visualization
./build/cpp-yolo26n detect --image test/bus.jpg --output benchmarks/bus_detected.png
./build/cpp-yolo26n detect --image test/zidane.jpg --output benchmarks/zidane_detected.png

# 4. Reproducible benchmark (100 warmup + 1,000 measured iterations, 4 threads)
./build/cpp-yolo26n benchmark --threads 4 --warmup 100 --iters 1000
```

---

## 4. Correctness Validation Results

- **Unit Tests (`ctest`)**: `1/1` (`cpp_unit_tests`: `4/4` assertions covering `computeLetterbox`, `preprocessRgbToNchw`, `inferOutputDescriptor` / `decodeOutput`, and `computeStats`) passed in both Debug and Release.
- **Model & Tensor Checks (`validate`)**:
  - Input validated: `[1, 3, 640, 640]` FP32
  - Output validated: `[1, 300, 6]` (`yolo26_end2end`)
  - 50 repeated inferences RSS check: `102,104 KiB -> 102,104 KiB` (`0 KiB` growth in Release; `101,620 KiB -> 101,620 KiB` in Debug)
  - Missing model file exits cleanly with non-zero status and ORT error message.
- **Exact Detection Parity (`bus.jpg` and `zidane.jpg`)**:
  - `test/bus.jpg` (`810×1080`, 5 detections):
    - `bus` `conf=0.9262` `box=[6.00, 228.84, 807.30, 749.18]` (identical to Zig)
    - `person` `conf=0.9161` `box=[46.64, 398.69, 237.18, 901.79]` (identical to Zig)
    - `person` `conf=0.8993` `box=[223.05, 405.37, 345.21, 863.08]` (identical to Zig)
    - `person` `conf=0.8614` `box=[668.60, 391.25, 810.00, 879.81]` (identical to Zig)
    - `person` `conf=0.5216` `box=[0.00, 553.48, 65.12, 874.03]` (identical to Zig)
  - `test/zidane.jpg` (`1280×720`, 3 detections):
    - `person` `conf=0.9199` `box=[745.16, 40.26, 1149.94, 710.62]` (identical to Zig)
    - `person` `conf=0.9024` `box=[119.70, 201.83, 984.82, 713.47]` (identical to Zig)
    - `tie` `conf=0.5083` `box=[433.69, 430.69, 525.48, 714.74]` (identical to Zig)

---

## 5. C++ Benchmark Trials (`100` warmup + `1,000` measured iterations, `threads=4`)

Three independent trials were executed sequentially (`cpp_benchmark_trial1.{csv,json}`, `cpp_benchmark_trial2.{csv,json}`, `cpp_benchmark_trial3.{csv,json}` / `cpp_benchmark_results.{csv,json}`):

| Metric | Trial 1 | Trial 2 | Trial 3 | 3-Trial Mean |
|---|---:|---:|---:|---:|
| App Startup Time (ms) | `92.764` | `51.239` | `57.257` | `67.087` |
| Model Load Time (ms) | `77.960` | `43.155` | `47.564` | `56.226` |
| Preprocessing Mean / Median / P95 / P99 (ms) | `3.509 / 3.700 / 4.195 / 4.495` | `3.827 / 3.815 / 4.742 / 5.921` | `4.041 / 4.265 / 4.975 / 5.477` | `3.792` |
| Inference (Pipeline) Mean / Median / P95 / P99 (ms) | `16.638 / 16.570 / 17.609 / 19.008` | `18.122 / 17.910 / 20.393 / 23.395` | `18.828 / 18.367 / 21.508 / 24.893` | `17.863` |
| Isolated Inference Mean / Median / P95 / P99 (ms) | `17.193 / 17.060 / 18.178 / 20.359` | `18.818 / 18.682 / 19.962 / 21.435` | `18.770 / 18.582 / 19.976 / 21.680` | `18.260` |
| Postprocessing Mean (ms) | `0.001` | `0.001` | `0.001` | `0.001` |
| **End-to-End Mean / Median / P95 / P99 (ms)** | **`20.148 / 20.020 / 21.443 / 22.827`** | **`21.951 / 21.675 / 24.932 / 29.556`** | **`22.870 / 22.429 / 26.324 / 29.640`** | **`21.656`** |
| **End-to-End Average FPS** | **`49.63`** | **`45.56`** | **`43.73`** | **`46.31`** |
| Peak RSS (`VmHWM`) | `105,268 KiB` (`102.80 MiB`) | `105,408 KiB` (`102.94 MiB`) | `105,372 KiB` (`102.90 MiB`) | `105,349 KiB` (`102.88 MiB`) |
| CPU Utilization (Pipeline / Inference-Only) | `397.40% / 397.54%` | `398.02% / 397.22%` | `397.85% / 398.03%` | `397.76% / 397.60%` |
| Binary Size (ELF bytes) | `329,544` | `329,544` | `329,544` | `329,544` |

*(Note: Trial 1 benefited from initial cold turbo boost, while Trials 2 and 3 settled into warm OS page cache for startup/model load and sustained thermal equilibrium for compute.)*

---

## 6. Direct Zig vs C++ Comparison

| Metric | Zig (`0.17.1`, `ReleaseFast`) | C++ (`g++ 16.2.1`, `-O3 -DNDEBUG`, 3-Trial Mean [Range]) | Difference (C++ vs Zig) |
|---|---:|---:|---:|
| App Startup Time | `58.631 ms` | `67.087 ms` [`51.239–92.764 ms`] | Parity (warm `51.2–57.3 ms`) |
| Model Load Time | `49.930 ms` | `56.226 ms` [`43.155–77.960 ms`] | Parity (warm `43.2–47.6 ms`) |
| Preprocessing (Mean) | **`3.418 ms`** | `3.792 ms` [`3.509–4.041 ms`] | Zig ~`9.9%` faster |
| Inference (Pipeline Mean) | `18.814 ms` | `17.863 ms` [`16.638–18.828 ms`] | Parity (within thermal variance) |
| Isolated Inference (Mean) | `19.732 ms` | `18.260 ms` [`17.193–18.818 ms`] | Parity (within thermal variance) |
| Postprocessing (Mean) | `0.001 ms` | `0.001 ms` [`0.00084–0.00100 ms`] | Identical (~`1 µs`) |
| End-to-End Latency (Mean) | `22.233 ms` | `21.656 ms` [`20.148–22.870 ms`] | Parity (`-2.6%`, within trial variance) |
| Average FPS | `44.98 FPS` | `46.31 FPS` [`43.73–49.63 FPS`] | Parity |
| CPU Utilization (Pipeline) | `397.40%` | `397.76%` [`397.40–398.02%`] | Identical (~4 full cores) |
| Peak RSS (`VmHWM`) | `107.57 MiB` (`110,152 KiB`) | **`102.88 MiB`** (`105,349 KiB`) | C++ `4.69 MiB` lower |
| Binary Size (unstripped ELF) | `5,532,648 bytes` (`text+data+bss`: `1,128,739`) | **`329,544 bytes`** (`text+data+bss`: `296,637`) | C++ ~`16.8×` smaller file (`3.8×` smaller sections) |

---

## 7. Explanation of Performance & Footprint Differences

1. **Inference Parity (`~17–19 ms`)**: Both Zig and C++ call the identical `libonnxruntime.so.1.29.0` C API (`OrtApi->Run`) using zero-copy preallocated `OrtValue` tensors and identical session threading (`4` intra-op, `1` inter-op). Inference compute time is therefore determined by ONNX Runtime's C++ MLAS/Eigen kernels and laptop CPU turbo/thermal state (`16.64 ms` cold boost vs `18.83 ms` sustained thermal equilibrium across consecutive trials).
2. **Preprocessing (`3.418 ms` Zig vs `3.792 ms` C++ mean)**: Zig's `b.standardTargetOptions(.{})` defaults to the host CPU (`native`, enabling AVX2/FMA vectorization on Intel Core 7 240H), whereas standard CMake `Release` (`-O3 -DNDEBUG`) targets baseline `x86-64` (SSE2) unless `-DENABLE_NATIVE_ARCH=ON` (`-march=native`) is enabled.
3. **Postprocessing (`~1 µs` in both)**: Decoding `[1, 300, 6]` requires scanning only 300 rows with zero heap allocations and no NMS, taking under `1 µs` in both languages.
4. **Peak RSS (`107.57 MiB` Zig vs `102.88 MiB` C++)**: The ~`37 MiB` warmup growth is identical in both (`~72 -> 110 MiB` in Zig, `~67 -> 105 MiB` in C++) because it comes from ONNX Runtime's CPU memory arena. The `~4.7 MiB` baseline difference reflects Zig's `std.Io.Threaded` / `init.gpa` runtime state and larger static `bss` (`262,901` bytes vs `64` bytes in C++).
5. **Binary Size (`5.53 MB` Zig vs `329.5 KB` C++)**: Zig `ReleaseFast` retains DWARF debug symbols by default (`with debug_info, not stripped`) and statically compiles `std` (`Io`, `fmt`, `sort`, stack-trace support), whereas CMake `Release` omits `-g` and dynamically links `libstdc++` and `libc`.
