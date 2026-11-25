# Fortran 2018 Edge Computer Vision Benchmark — YOLO26n (ONNX Runtime CPU)

Controlled **Modern Fortran (Fortran 2018 / GFortran 16.2.1)** implementation of the **YOLO26n (FP32, 640×640)** edge computer vision benchmark using the **ONNX Runtime 1.29.0 C API** via `ISO_C_BINDING` on Linux `x86_64` CPU.

---

## 1. Directory Structure

```
fortran-human-detection/
├── src/
│   ├── main.f90             # CLI (detect, validate, benchmark, kernel-bench)
│   ├── detector.f90         # Preallocated Fortran TARGET tensor lifecycle
│   ├── onnx_runtime.f90     # ISO_C_BINDING interface to ONNX Runtime C API
│   ├── image.f90            # RGB24 view (3, width, height) & stb_image bridge
│   ├── preprocessing.f90    # Deterministic letterbox + bilinear resize + NCHW + kernels
│   ├── postprocessing.f90   # YOLO26 [1, 300, 6] & raw fallback decoders + NMS
│   ├── visualization.f90    # Pure-Fortran bounding-box & 5x7 bitmap text drawing
│   ├── benchmark.f90        # 100-warmup + 1,000-iter pipeline & kernel benchmarks
│   ├── metrics.f90          # Monotonic timer stats, RSS, CPU utilization
│   └── unit_tests.f90       # CTest unit test suite
├── c/
│   ├── onnx_bridge.c        # Minimal ORT C API wrapper (OrtApi vtable + zero-copy bind)
│   ├── image_bridge.c       # stb_image / stb_image_write wrapper
│   └── system_metrics.c     # CLOCK_MONOTONIC, CLOCK_PROCESS_CPUTIME_ID, /proc/self/status
├── models/
│   └── yolov26n.onnx        # Identical ONNX model (SHA-256: 8ef199e41e23b930...)
├── include/                 # ONNX Runtime 1.29.0 C API + stb_image headers
├── lib/                     # libonnxruntime.so symlinks
├── test/                    # Fixed test images (bus.jpg, zidane.jpg)
├── benchmarks/              # Fortran trial CSV/JSON reports & annotated PNGs
├── CMakeLists.txt
└── README.md
```

---

## 2. Modern Fortran Architecture & Memory Ownership

- **Zero-Copy Column-Major Alignment**:
  - `stb_image` returns C row-major RGB24 `(height, width, 3)`. Wrapping the `c_ptr` with `c_f_pointer(c_buf, view, [3, width, height])` maps directly to Fortran column-major `(3, width, height)` with **zero transpose or copy**, converting signed `integer(c_int8_t)` bytes to unsigned `[0..255]` via `iand(int(b, int32), 255)`.
  - `det%input_tensor(640, 640, 3)` and `det%output_tensor(6, 300)` have the `TARGET` attribute and column-major memory layouts identical to C row-major `[1, 3, 640, 640]` (NCHW) and `[1, 300, 6]` respectively.
  - `c_loc(det%input_tensor(1, 1, 1))` and `c_loc(det%output_tensor(1, 1))` are bound once to ONNX Runtime via `CreateTensorWithDataAsOrtValue`.
  - **Lifecycle Contract**: `detector_destroy` calls `ort_destroy_session` (releasing all `OrtValue`s and `OrtSession`) **before** deallocating the Fortran `TARGET` arrays.

---

## 3. Build, Compiler Optimization & Vectorization

- **Compiler**: `GNU Fortran (GCC) 16.2.1 20260810` (`-cpp -std=f2018 -Wall -Wextra`)
- **Debug Build**: `-g -O0 -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`
- **Release Build**: `-O3 -march=native -mtune=native -DNDEBUG` (no unsafe `-ffast-math` flags, preserving numerical equivalence)

### Vectorization Report (`-fopt-info-vec-optimized`)

GFortran `16.2.1` vectorized the inner `dx` bilinear preprocessing loops and tensor padding fill in `src/preprocessing.f90` with **32-byte AVX2 vectors (`unroll factor 8`)** and **16-byte SSE epilogues (`unroll factor 4`)**.

### Commands

```bash
cd fortran-human-detection

# 1. Debug build + bounds/FPE checks + CTest + validate
cmake -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ctest --test-dir build-debug --output-on-failure
./build-debug/fortran-yolo26n validate

# 2. Release build (-O3 -march=native -mtune=native -DNDEBUG)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/fortran-yolo26n validate

# 3. Single-image detection + visualization
./build/fortran-yolo26n detect --image test/bus.jpg --output benchmarks/bus_detected.png
./build/fortran-yolo26n detect --image test/zidane.jpg --output benchmarks/zidane_detected.png

# 4. Numerical kernel benchmark (Section 10)
./build/fortran-yolo26n kernel-bench --image test/bus.jpg --warmup 100 --iters 500
./build/fortran-yolo26n kernel-bench --image test/zidane.jpg --warmup 100 --iters 500

# 5. Full detector benchmark (100 warmup + 1,000 measured iterations, 4 threads)
./build/fortran-yolo26n benchmark --threads 4 --warmup 100 --iters 1000
```

---

## 4. Validation Results

- **CTest (`fortran_unit_tests`)**: `1/1` passed (`4/4` assertions) in both Debug and Release.
- **Model & Tensor Checks (`validate`)**:
  - Input `[1, 3, 640, 640]` FP32, Output `[1, 300, 6]` (`yolo26_end2end`)
  - 50 repeated inferences RSS check: `104,476 KiB -> 104,476 KiB` (`0 KiB` growth in Release; `104,436 -> 104,436 KiB` in Debug)
  - Missing model file exits non-zero (`STOP 1`) cleanly with ONNX Runtime error message.
- **Detection Parity (`bus.jpg` and `zidane.jpg`)**:
  - `test/bus.jpg` (5 detections): `bus` `.9262 [6.00, 228.84, 807.30, 749.18]`, `person` `.9161 [46.64, 398.69, 237.18, 901.79]`, `person` `.8993 [223.05, 405.37, 345.21, 863.08]`, `person` `.8614 [668.60, 391.25, 810.00, 879.81]`, `person` `.5216 [.00, 553.48, 65.12, 874.03]` — exact match with Zig and C++.
  - `test/zidane.jpg` (3 detections): `person` `.9199 [745.16, 40.26, 1149.94, 710.62]`, `person` `.9024 [119.70, 201.83, 984.82, 713.47]`, `tie` `.5083 [433.69, 430.69, 525.48, 714.74]` — exact match with Zig and C++.

---

## 5. Numerical Kernel Benchmark (`kernel-bench`, 100 warmup + 500 measured)

| Kernel | `bus.jpg` (`810×1080 -> 480×640`) Mean / Median (ms) | `zidane.jpg` (`1280×720 -> 640×360`) Mean / Median (ms) |
|---|---:|---:|
| 1. RGB `u8 -> f32` Normalization | `0.885 / 0.823` | `0.892 / 0.845` |
| 2. Bilinear Resize (`f32`) | `1.727 / 1.652` | `1.282 / 1.244` |
| 3. Letterbox Padding (`640×640`) | `0.773 / 0.696` | `0.637 / 0.602` |
| 4. Layout Conversion (`HWC -> CHW`) | `0.520 / 0.496` | `0.518 / 0.505` |
| **5. Fused Single-Thread Preprocess** | **`2.682 / 2.608`** (`372.79 FPS`) | **`2.033 / 2.013`** (`491.80 FPS`) |
| **6. Fused OpenMP Preprocess (`!$omp parallel do`)** | **`0.721 / 0.697`** (`1,387.78 FPS`) | **`0.585 / 0.583`** (`1,709.01 FPS`) |

*(Note: For fairness against Zig and C++, the full detector benchmark uses kernel #5 single-threaded fused preprocessing.)*

---

## 6. Three-Way Controlled Comparison: Zig vs C++ vs Fortran

Executed in **9 alternating trials** (`Zig -> C++ -> Fortran` × 3, `100` warmup + `1,000` measured iterations, `threads=4`, identical `libonnxruntime.so.1.29.0`, identical external `/proc/<pid>/status` `VmRSS` sampling every `20 ms`):

### Performance, Memory, and CPU (Mean ± SD across 3 trials)

| Metric | Zig `0.17.1` (`ReleaseFast`) | C++ `g++ 16.2.1` (`-O3 -march=native`) | Fortran `gfortran 16.2.1` (`-O3 -march=native`) |
|---|---:|---:|---:|
| App Startup (ms) | `64.063 ± 5.758` | `59.330 ± 2.733` | `59.088 ± 1.532` |
| Model Load (ms) | `52.998 ± 3.271` | `49.446 ± 2.696` | `49.437 ± 1.218` |
| **Preprocessing Mean (ms)** | `3.829 ± 0.222` | `3.358 ± 0.040` | **`3.029 ± 0.050`** |
| Pipeline Inference Mean (ms) | `20.429 ± 0.951` | `20.908 ± 0.311` | `21.180 ± 0.439` |
| Isolated Inference Mean (ms) | `21.069 ± 0.658` | `21.435 ± 0.451` | `21.537 ± 0.528` |
| Postprocessing Mean (ms) | `0.001 ± 0.000` | `0.001 ± 0.000` | `0.002 ± 0.000` |
| **End-to-End Mean (ms)** | `24.259 ± 1.173` | `24.266 ± 0.351` | `24.211 ± 0.489` |
| End-to-End Median (ms) | `24.027 ± 1.176` | `24.044 ± 0.397` | `23.942 ± 0.559` |
| End-to-End P95 (ms) | `26.579 ± 1.116` | `26.621 ± 0.090` | `26.999 ± 0.716` |
| End-to-End P99 (ms) | `29.883 ± 1.396` | `29.601 ± 0.532` | `30.299 ± 1.712` |
| **End-to-End Avg FPS** | `41.29 ± 2.05` | `41.22 ± 0.59` | `41.32 ± 0.83` |
| CPU Utilization (Pipeline) | `396.81 ± 0.55%` | `397.04 ± 0.24%` | `396.70 ± 0.41%` |
| **Peak RSS (`VmHWM`)** | `108,125 ± 20 KiB` (`105.59 MiB`) | **`106,776 ± 45 KiB`** (`104.27 MiB`) | `107,712 ± 17 KiB` (`105.19 MiB`) |
| **Average RSS (`20 ms`)** | `108,002 ± 25 KiB` (`105.47 MiB`) | **`106,656 ± 49 KiB`** (`104.16 MiB`) | `107,584 ± 29 KiB` (`105.06 MiB`) |

### Binary Size & ELF Comparison

| Metric | Zig | C++ | Fortran |
|---|---:|---:|---:|
| Original ELF (bytes) | `5,713,768` | **`366,664`** | `438,168` |
| Stripped ELF (`strip --strip-all`) | `910,688` | **`343,120`** | `413,304` |
| `text / data / bss` (bytes) | `877,158 / 30,376 / 266,261` | `331,144 / 2,704 / 64` | `404,293 / 3,088 / 400` |
| Dynamic `NEEDED` Libraries | `libonnxruntime.so.1`, `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2` | `libonnxruntime.so.1`, `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` | `libonnxruntime.so.1`, `libgomp.so.1`, `libgfortran.so.5`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` |

### Key Conclusions

1. **Preprocessing Winner — Fortran (`3.029 ms`)**: Fortran is **9.8% faster than C++ (`3.358 ms`, >6× SD)** and **20.9% faster than Zig (`3.829 ms`)** on the single-threaded bilinear preprocessing kernel due to non-aliasing array semantics and GFortran's 32-byte AVX2 unroll-8 vectorization over contiguous column-major `(W, H, C)` slices. With OpenMP, Fortran reaches `0.59–0.72 ms`.
2. **End-to-End Throughput — Statistical Tie (`~41.2–41.3 FPS`)**: Total end-to-end latency (`24.21–24.27 ms`) is statistically indistinguishable across Zig, C++, and Fortran because `>85%` of runtime occurs inside ONNX Runtime's C++ compute kernels.
3. **Memory Footprint — C++ Lowest (`104.27 MiB` Peak RSS)**: C++ uses `~0.92 MiB` less RSS than Fortran (due to `libgfortran`/`libgomp`) and `~1.32 MiB` less than Zig.
4. **Binary Size — C++ Smallest (`343 KB` stripped), Fortran Close Second (`413 KB` stripped)**: Both dynamically link their language runtimes, whereas Zig (`911 KB` stripped) statically links `std` without depending on `libstdc++` or `libgfortran`.
