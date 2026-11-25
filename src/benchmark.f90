module benchmark_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    use image_mod
    use preprocessing_mod
    use postprocessing_mod
    use detector_mod
    use metrics_mod
    implicit none
    private

    public :: benchmark_config_t, benchmark_report_t
    public :: run_benchmark, run_kernel_benchmark

    type :: benchmark_config_t
        character(len=512) :: model_path = "models/yolov26n.onnx"
        character(len=512) :: image_paths(16) = ""
        integer :: image_count = 0
        integer :: warmup_iters = 100
        integer :: measured_iters = 1000
        integer(int32) :: threads = 4
        real(real32) :: conf_threshold = 0.25_real32
        real(real32) :: iou_threshold = 0.45_real32
        logical :: force_nms = .false.
        character(len=512) :: csv_out_path = "benchmarks/fortran_benchmark_results.csv"
        character(len=512) :: json_out_path = "benchmarks/fortran_benchmark_results.json"
    end type benchmark_config_t

    type :: benchmark_report_t
        real(real64) :: app_startup_ms = 0.0_real64
        real(real64) :: model_load_ms = 0.0_real64
        integer(int64) :: binary_size_bytes = 0
        integer(int64) :: peak_rss_kb = 0
        integer(int64) :: rss_before_kb = 0
        integer(int64) :: rss_after_kb = 0
        real(real64) :: pipeline_cpu_util_pct = 0.0_real64
        real(real64) :: infer_only_cpu_util_pct = 0.0_real64
        type(latency_stats_t) :: preprocessing
        type(latency_stats_t) :: pipeline_inference
        type(latency_stats_t) :: postprocessing
        type(latency_stats_t) :: total_e2e
        type(latency_stats_t) :: inference_only
        integer(int32) :: last_detections_count = 0
    end type benchmark_report_t

contains

    pure function optimize_mode_str() result(s)
        character(len=:), allocatable :: s
#ifdef NDEBUG
        s = "release"
#else
        s = "debug"
#endif
    end function optimize_mode_str

    function fmt_f6(val) result(s)
        real(real64), intent(in) :: val
        character(len=:), allocatable :: s
        character(len=64) :: buf
        write(buf, '(F0.6)') val
        buf = adjustl(buf)
        if (buf(1:1) == '.') then
            s = '0' // trim(buf)
        else if (buf(1:2) == '-.') then
            s = '-0.' // trim(buf(3:))
        else
            s = trim(buf)
        end if
    end function fmt_f6

    function fmt_f4(val) result(s)
        real(real64), intent(in) :: val
        character(len=:), allocatable :: s
        character(len=64) :: buf
        write(buf, '(F0.4)') val
        buf = adjustl(buf)
        if (buf(1:1) == '.') then
            s = '0' // trim(buf)
        else if (buf(1:2) == '-.') then
            s = '-0.' // trim(buf(3:))
        else
            s = trim(buf)
        end if
    end function fmt_f4

    subroutine print_stats_row(label, s)
        character(len=*), intent(in) :: label
        type(latency_stats_t), intent(in) :: s
        write(*, '(A22," | ",F7.3," | ",F7.3," | ",F7.3," | ",F7.3," | ",F7.3," | ",F7.3," | ",F7.2)') &
            label, s%mean_ms, s%median_ms, s%p95_ms, s%p99_ms, s%min_ms, s%max_ms, s%avg_fps
    end subroutine print_stats_row

    subroutine print_report(det, cfg, r, compiler_ver)
        type(detector_t), intent(in) :: det
        type(benchmark_config_t), intent(in) :: cfg
        type(benchmark_report_t), intent(in) :: r
        character(len=*), intent(in) :: compiler_ver
        character(len=:), allocatable :: fmt_name, opt_mode

        fmt_name = output_format_name(det%output_desc%format)
        opt_mode = optimize_mode_str()

        write(*, '(A)') ""
        write(*, '(A)') "====================================================================="
        write(*, '(A)') " Fortran Edge Computer Vision Benchmark — YOLO26n (ONNX Runtime CPU)"
        write(*, '(A)') "====================================================================="
        write(*, '(A,A," (optimize=",A,")")') "Compiler            : ", trim(compiler_ver), opt_mode
        write(*, '(A,A," (CPUExecutionProvider, threads=",I0,")")') "ONNX Runtime        : ", &
            trim(det%session%ort_version), cfg%threads
        write(*, '(A,A)') "Model               : ", trim(cfg%model_path)
        write(*, '(A,"[",I0,", ",I0,", ",I0,", ",I0,"] (FP32)")') "Input Tensor        : ", &
            det%session%input_dims(1), det%session%input_dims(2), det%session%input_dims(3), det%session%input_dims(4)
        write(*, '(A,"[",I0,", ",I0,", ",I0,"] (",A,")")') "Output Tensor       : ", &
            det%session%output_dims(1), det%session%output_dims(2), det%session%output_dims(3), fmt_name
        write(*, '(A,I0," warmup + ",I0," measured (",I0," fixed image(s))")') "Iterations          : ", &
            cfg%warmup_iters, cfg%measured_iters, cfg%image_count
        write(*, '(A)') "---------------------------------------------------------------------"
        write(*, '(A)') "Startup & Resources:"
        write(*, '(A,F0.3," ms")') "  App Startup Time  : ", r%app_startup_ms
        write(*, '(A,F0.3," ms")') "  Model Load Time   : ", r%model_load_ms
        write(*, '(A,I0," bytes (",F0.2," KiB)")') "  Binary Size       : ", &
            r%binary_size_bytes, real(r%binary_size_bytes, real64) / 1024.0_real64
        write(*, '(A,I0," KiB (",F0.2," MiB)")') "  Peak RSS (VmHWM)  : ", &
            r%peak_rss_kb, real(r%peak_rss_kb, real64) / 1024.0_real64
        write(*, '(A,I0," KiB -> ",I0," KiB (delta=",I0," KiB)")') "  RSS Before/After  : ", &
            r%rss_before_kb, r%rss_after_kb, r%rss_after_kb - r%rss_before_kb
        write(*, '(A,F0.2,"% (pipeline), ",F0.2,"% (inference-only) [process CPU/wall]")') &
            "  CPU Utilization   : ", r%pipeline_cpu_util_pct, r%infer_only_cpu_util_pct
        write(*, '(A)') "---------------------------------------------------------------------"
        write(*, '(A)') "Stage Latencies (ms)   |    Mean |  Median |     P95 |     P99 |     Min |     Max |     FPS"
        call print_stats_row("Preprocessing", r%preprocessing)
        call print_stats_row("Inference (Pipeline)", r%pipeline_inference)
        call print_stats_row("Postprocessing", r%postprocessing)
        call print_stats_row("Total End-to-End", r%total_e2e)
        call print_stats_row("Inference-Only (Isol)", r%inference_only)
        write(*, '(A)') "====================================================================="
        write(*, '(A)') ""
    end subroutine print_report

    subroutine write_csv_report(det, cfg, r, compiler_ver)
        type(detector_t), intent(in) :: det
        type(benchmark_config_t), intent(in) :: cfg
        type(benchmark_report_t), intent(in) :: r
        character(len=*), intent(in) :: compiler_ver

        integer :: u, i, ios
        character(len=:), allocatable :: opt_mode
        character(len=24) :: stage_names(5)
        type(latency_stats_t) :: stage_stats(5)
        real(real64) :: stage_cpu(5)

        opt_mode = optimize_mode_str()
        stage_names = [character(len=24) :: &
            "preprocessing", "pipeline_inference", "postprocessing", "total_e2e", "inference_only"]
        stage_stats = [r%preprocessing, r%pipeline_inference, r%postprocessing, r%total_e2e, r%inference_only]
        stage_cpu = [ &
            r%pipeline_cpu_util_pct, r%pipeline_cpu_util_pct, r%pipeline_cpu_util_pct, &
            r%pipeline_cpu_util_pct, r%infer_only_cpu_util_pct]

        open(newunit=u, file=trim(cfg%csv_out_path), status="replace", action="write", iostat=ios)
        if (ios /= 0) then
            write(*, '(A,A)') "Failed to write CSV: ", trim(cfg%csv_out_path)
            stop 1
        end if

        write(u, '(A)') "language,compiler_version,optimize_mode,ort_version,threads,warmup_iters,measured_iters," // &
            "stage,count,mean_ms,median_ms,p95_ms,p99_ms,min_ms,max_ms,std_dev_ms,avg_fps," // &
            "model_load_ms,app_startup_ms,peak_rss_kb,cpu_util_pct,binary_size_bytes"

        do i = 1, 5
            write(u, '("fortran,",A,",",A,",",A,",",I0,",",I0,",",I0,",",A,",",I0,",",' // &
                'A,",",A,",",A,",",A,",",A,",",A,",",A,",",A,",",A,",",A,",",I0,",",A,",",I0)') &
                trim(compiler_ver), opt_mode, trim(det%session%ort_version), &
                cfg%threads, cfg%warmup_iters, cfg%measured_iters, &
                trim(stage_names(i)), stage_stats(i)%count, &
                fmt_f6(stage_stats(i)%mean_ms), fmt_f6(stage_stats(i)%median_ms), &
                fmt_f6(stage_stats(i)%p95_ms), fmt_f6(stage_stats(i)%p99_ms), &
                fmt_f6(stage_stats(i)%min_ms), fmt_f6(stage_stats(i)%max_ms), &
                fmt_f6(stage_stats(i)%std_dev_ms), fmt_f4(stage_stats(i)%avg_fps), &
                fmt_f6(r%model_load_ms), fmt_f6(r%app_startup_ms), &
                r%peak_rss_kb, fmt_f4(stage_cpu(i)), r%binary_size_bytes
        end do
        close(u)
    end subroutine write_csv_report

    subroutine write_json_stage(u, name, s, trailing_comma)
        integer, intent(in) :: u
        character(len=*), intent(in) :: name
        type(latency_stats_t), intent(in) :: s
        logical, intent(in) :: trailing_comma

        write(u, '(A,A,A)') '    "', name, '": {'
        write(u, '(A,I0,A)') '      "count": ', s%count, ','
        write(u, '(A,A,A)') '      "mean_ms": ', fmt_f6(s%mean_ms), ','
        write(u, '(A,A,A)') '      "median_ms": ', fmt_f6(s%median_ms), ','
        write(u, '(A,A,A)') '      "p95_ms": ', fmt_f6(s%p95_ms), ','
        write(u, '(A,A,A)') '      "p99_ms": ', fmt_f6(s%p99_ms), ','
        write(u, '(A,A,A)') '      "min_ms": ', fmt_f6(s%min_ms), ','
        write(u, '(A,A,A)') '      "max_ms": ', fmt_f6(s%max_ms), ','
        write(u, '(A,A,A)') '      "std_dev_ms": ', fmt_f6(s%std_dev_ms), ','
        write(u, '(A,A)') '      "avg_fps": ', fmt_f4(s%avg_fps)
        if (trailing_comma) then
            write(u, '(A)') '    },'
        else
            write(u, '(A)') '    }'
        end if
    end subroutine write_json_stage

    subroutine write_json_report(det, cfg, r, compiler_ver)
        type(detector_t), intent(in) :: det
        type(benchmark_config_t), intent(in) :: cfg
        type(benchmark_report_t), intent(in) :: r
        character(len=*), intent(in) :: compiler_ver

        integer :: u, ios
        character(len=:), allocatable :: opt_mode, fmt_name

        opt_mode = optimize_mode_str()
        fmt_name = output_format_name(det%output_desc%format)

        open(newunit=u, file=trim(cfg%json_out_path), status="replace", action="write", iostat=ios)
        if (ios /= 0) then
            write(*, '(A,A)') "Failed to write JSON: ", trim(cfg%json_out_path)
            stop 1
        end if

        write(u, '(A)') '{'
        write(u, '(A)') '  "language": "fortran",'
        write(u, '(A,A,A)') '  "compiler_version": "', trim(compiler_ver), '",'
        write(u, '(A,A,A)') '  "optimize_mode": "', opt_mode, '",'
        write(u, '(A,A,A)') '  "onnxruntime_version": "', trim(det%session%ort_version), '",'
        write(u, '(A)') '  "execution_provider": "CPUExecutionProvider",'
        write(u, '(A,A,A)') '  "model_path": "', trim(cfg%model_path), '",'
        write(u, '(A,I0,", ",I0,", ",I0,", ",I0,A)') '  "input_shape": [', &
            det%session%input_dims(1), det%session%input_dims(2), det%session%input_dims(3), det%session%input_dims(4), '],'
        write(u, '(A,I0,", ",I0,", ",I0,A)') '  "output_shape": [', &
            det%session%output_dims(1), det%session%output_dims(2), det%session%output_dims(3), '],'
        write(u, '(A,A,A)') '  "output_format": "', fmt_name, '",'
        write(u, '(A)') '  "config": {'
        write(u, '(A,I0,A)') '    "threads": ', cfg%threads, ','
        write(u, '(A,I0,A)') '    "warmup_iterations": ', cfg%warmup_iters, ','
        write(u, '(A,I0,A)') '    "measured_iterations": ', cfg%measured_iters, ','
        write(u, '(A,A,A)') '    "conf_threshold": ', fmt_f4(real(cfg%conf_threshold, real64)), ','
        write(u, '(A,A,A)') '    "iou_threshold": ', fmt_f4(real(cfg%iou_threshold, real64)), ','
        write(u, '(A,I0)') '    "image_count": ', cfg%image_count
        write(u, '(A)') '  },'
        write(u, '(A)') '  "system_metrics": {'
        write(u, '(A,A,A)') '    "app_startup_ms": ', fmt_f6(r%app_startup_ms), ','
        write(u, '(A,A,A)') '    "model_load_ms": ', fmt_f6(r%model_load_ms), ','
        write(u, '(A,I0,A)') '    "binary_size_bytes": ', r%binary_size_bytes, ','
        write(u, '(A,I0,A)') '    "peak_rss_kb": ', r%peak_rss_kb, ','
        write(u, '(A,I0,A)') '    "rss_before_kb": ', r%rss_before_kb, ','
        write(u, '(A,I0,A)') '    "rss_after_kb": ', r%rss_after_kb, ','
        write(u, '(A,A,A)') '    "pipeline_cpu_utilization_pct": ', fmt_f4(r%pipeline_cpu_util_pct), ','
        write(u, '(A,A)') '    "inference_only_cpu_utilization_pct": ', fmt_f4(r%infer_only_cpu_util_pct)
        write(u, '(A)') '  },'
        write(u, '(A)') '  "stages": {'

        call write_json_stage(u, "preprocessing", r%preprocessing, .true.)
        call write_json_stage(u, "pipeline_inference", r%pipeline_inference, .true.)
        call write_json_stage(u, "postprocessing", r%postprocessing, .true.)
        call write_json_stage(u, "total_end_to_end", r%total_e2e, .true.)
        call write_json_stage(u, "inference_only", r%inference_only, .false.)

        write(u, '(A)') '  }'
        write(u, '(A)') '}'
        close(u)
    end subroutine write_json_report

    function run_benchmark(app_start_ns, cfg) result(report)
        integer(int64), intent(in) :: app_start_ns
        type(benchmark_config_t), intent(in) :: cfg
        type(benchmark_report_t) :: report

        type(image_t), allocatable :: images(:)
        type(detector_t) :: det
        type(detector_config_t) :: dcfg
        type(letterbox_params_t) :: lb
        type(process_memory_t) :: rss_before, rss_after
        character(len=512) :: err_msg
        character(len=64) :: compiler_ver
        integer :: i, img_idx
        integer(int32) :: det_count = 0
        integer(int64), volatile :: sink
        integer(int64) :: load_start, load_end, ready_ns
        integer(int64) :: pipe_wall_start, pipe_wall_end, pipe_cpu_start, pipe_cpu_end
        integer(int64) :: inf_wall_start, inf_wall_end, inf_cpu_start, inf_cpu_end
        integer(int64) :: t0, t1, t2, t3
        integer(int64), allocatable :: pre_ns(:), inf_ns(:), post_ns(:), tot_ns(:), inf_only_ns(:), scratch_ns(:)

        if (cfg%image_count <= 0) then
            write(*, '(A)') "Error: No input images provided"
            stop 1
        end if

        write(compiler_ver, '("gfortran ",I0,".",I0,".",I0)') __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__

        allocate(images(cfg%image_count))
        do i = 1, cfg%image_count
            if (.not. image_load(cfg%image_paths(i), images(i))) then
                write(*, '(A,A)') "Failed to load image: ", trim(cfg%image_paths(i))
                stop 1
            end if
        end do

        dcfg = detector_config_t(cfg%model_path, cfg%threads, 1_int32, cfg%conf_threshold, cfg%iou_threshold, cfg%force_nms)
        load_start = monotonic_now_ns()
        if (.not. detector_init(det, dcfg, err_msg)) then
            write(*, '(A,A)') "Detector init failed: ", trim(err_msg)
            stop 1
        end if
        load_end = monotonic_now_ns()
        ready_ns = monotonic_now_ns()

        report%model_load_ms = real(load_end - load_start, real64) / 1000000.0_real64
        report%app_startup_ms = real(ready_ns - app_start_ns, real64) / 1000000.0_real64

        allocate(pre_ns(cfg%measured_iters))
        allocate(inf_ns(cfg%measured_iters))
        allocate(post_ns(cfg%measured_iters))
        allocate(tot_ns(cfg%measured_iters))
        allocate(inf_only_ns(cfg%measured_iters))
        allocate(scratch_ns(cfg%measured_iters))

        rss_before = read_process_memory()
        sink = 0

        ! Pass 1: Complete Pipeline
        do i = 1, cfg%warmup_iters
            img_idx = mod(i - 1, cfg%image_count) + 1
            lb = detector_preprocess(det, images(img_idx)%view, images(img_idx)%width, images(img_idx)%height)
            if (.not. detector_infer_raw(det, err_msg)) then
                write(*, '(A,A)') "Inference error: ", trim(err_msg)
                stop 1
            end if
            det_count = detector_postprocess(det, lb)
            sink = sink + det_count
        end do

        pipe_wall_start = monotonic_now_ns()
        pipe_cpu_start = process_cpu_now_ns()

        do i = 1, cfg%measured_iters
            img_idx = mod(i - 1, cfg%image_count) + 1

            t0 = monotonic_now_ns()
            lb = detector_preprocess(det, images(img_idx)%view, images(img_idx)%width, images(img_idx)%height)
            t1 = monotonic_now_ns()
            if (.not. detector_infer_raw(det, err_msg)) then
                write(*, '(A,A)') "Inference error: ", trim(err_msg)
                stop 1
            end if
            t2 = monotonic_now_ns()
            det_count = detector_postprocess(det, lb)
            t3 = monotonic_now_ns()

            sink = sink + det_count
            pre_ns(i) = t1 - t0
            inf_ns(i) = t2 - t1
            post_ns(i) = t3 - t2
            tot_ns(i) = t3 - t0
        end do

        pipe_cpu_end = process_cpu_now_ns()
        pipe_wall_end = monotonic_now_ns()

        ! Pass 2: Inference-Only Benchmark
        lb = detector_preprocess(det, images(1)%view, images(1)%width, images(1)%height)
        do i = 1, cfg%warmup_iters
            if (.not. detector_infer_raw(det, err_msg)) then
                write(*, '(A,A)') "Inference error: ", trim(err_msg)
                stop 1
            end if
        end do

        inf_wall_start = monotonic_now_ns()
        inf_cpu_start = process_cpu_now_ns()

        do i = 1, cfg%measured_iters
            t0 = monotonic_now_ns()
            if (.not. detector_infer_raw(det, err_msg)) then
                write(*, '(A,A)') "Inference error: ", trim(err_msg)
                stop 1
            end if
            t1 = monotonic_now_ns()
            inf_only_ns(i) = t1 - t0
        end do

        inf_cpu_end = process_cpu_now_ns()
        inf_wall_end = monotonic_now_ns()

        rss_after = read_process_memory()
        report%binary_size_bytes = read_executable_size_bytes()
        report%peak_rss_kb = rss_after%peak_rss_kb
        report%rss_before_kb = rss_before%current_rss_kb
        report%rss_after_kb = rss_after%current_rss_kb
        report%pipeline_cpu_util_pct = compute_cpu_utilization_pct(pipe_cpu_end - pipe_cpu_start, pipe_wall_end - pipe_wall_start)
        report%infer_only_cpu_util_pct = compute_cpu_utilization_pct(inf_cpu_end - inf_cpu_start, inf_wall_end - inf_wall_start)
        report%preprocessing = compute_stats(pre_ns, scratch_ns)
        report%pipeline_inference = compute_stats(inf_ns, scratch_ns)
        report%postprocessing = compute_stats(post_ns, scratch_ns)
        report%total_e2e = compute_stats(tot_ns, scratch_ns)
        report%inference_only = compute_stats(inf_only_ns, scratch_ns)
        report%last_detections_count = det_count

        if (sink < 0) write(*, *) sink

        call print_report(det, cfg, report, compiler_ver)
        call write_csv_report(det, cfg, report, compiler_ver)
        call write_json_report(det, cfg, report, compiler_ver)

        call detector_destroy(det)
        do i = 1, cfg%image_count
            call image_destroy(images(i))
        end do
        deallocate(images, pre_ns, inf_ns, post_ns, tot_ns, inf_only_ns, scratch_ns)
    end function run_benchmark

    ! Section 10: Standalone Numerical Kernel Benchmark (RGB norm, bilinear, letterbox, HWC->CHW, fused, OpenMP)
    subroutine run_kernel_benchmark(image_path, warmup_iters, measured_iters)
        character(len=*), intent(in) :: image_path
        integer, intent(in) :: warmup_iters, measured_iters

        type(image_t) :: img
        type(letterbox_params_t) :: lb
        real(c_float), allocatable :: src_f32(:, :, :), resized_f32(:, :, :), padded_hwc(:, :, :), chw_tensor(:, :, :)
        integer(int64), allocatable :: t_norm(:), t_resize(:), t_pad(:), t_layout(:), t_fused(:), t_omp(:), scratch(:)
        integer(int64) :: t0, t1
        real(c_float), volatile :: sink
        type(latency_stats_t) :: st_norm, st_resize, st_pad, st_layout, st_fused, st_omp
        integer :: i

        if (.not. image_load(image_path, img)) then
            write(*, '(A,A)') "Failed to load image: ", trim(image_path)
            stop 1
        end if

        lb = compute_letterbox(img%width, img%height, INPUT_WIDTH, INPUT_HEIGHT)

        allocate(src_f32(3, img%width, img%height))
        allocate(resized_f32(3, lb%resized_w, lb%resized_h))
        allocate(padded_hwc(3, INPUT_WIDTH, INPUT_HEIGHT))
        allocate(chw_tensor(INPUT_WIDTH, INPUT_HEIGHT, 3))

        allocate(t_norm(measured_iters), t_resize(measured_iters), t_pad(measured_iters))
        allocate(t_layout(measured_iters), t_fused(measured_iters), t_omp(measured_iters), scratch(measured_iters))

        sink = 0.0_c_float

        do i = 1, warmup_iters
            call kernel_rgb_normalize(img%view, img%width, img%height, src_f32)
            call kernel_bilinear_resize_rgb(src_f32, img%width, img%height, lb%resized_w, lb%resized_h, resized_f32)
            call kernel_letterbox_pad(resized_f32, lb%resized_w, lb%resized_h, lb%pad_x, lb%pad_y, padded_hwc)
            call kernel_hwc_to_chw(padded_hwc, INPUT_WIDTH, INPUT_HEIGHT, chw_tensor)
            lb = preprocess_rgb_to_nchw(img%view, img%width, img%height, INPUT_WIDTH, INPUT_HEIGHT, chw_tensor)
            lb = preprocess_rgb_to_nchw_omp(img%view, img%width, img%height, INPUT_WIDTH, INPUT_HEIGHT, chw_tensor)
            sink = sink + chw_tensor(1, 1, 1)
        end do

        do i = 1, measured_iters
            t0 = monotonic_now_ns()
            call kernel_rgb_normalize(img%view, img%width, img%height, src_f32)
            t1 = monotonic_now_ns()
            t_norm(i) = t1 - t0
            sink = sink + src_f32(1, 1, 1)

            t0 = monotonic_now_ns()
            call kernel_bilinear_resize_rgb(src_f32, img%width, img%height, lb%resized_w, lb%resized_h, resized_f32)
            t1 = monotonic_now_ns()
            t_resize(i) = t1 - t0
            sink = sink + resized_f32(1, 1, 1)

            t0 = monotonic_now_ns()
            call kernel_letterbox_pad(resized_f32, lb%resized_w, lb%resized_h, lb%pad_x, lb%pad_y, padded_hwc)
            t1 = monotonic_now_ns()
            t_pad(i) = t1 - t0
            sink = sink + padded_hwc(1, 1, 1)

            t0 = monotonic_now_ns()
            call kernel_hwc_to_chw(padded_hwc, INPUT_WIDTH, INPUT_HEIGHT, chw_tensor)
            t1 = monotonic_now_ns()
            t_layout(i) = t1 - t0
            sink = sink + chw_tensor(1, 1, 1)

            t0 = monotonic_now_ns()
            lb = preprocess_rgb_to_nchw(img%view, img%width, img%height, INPUT_WIDTH, INPUT_HEIGHT, chw_tensor)
            t1 = monotonic_now_ns()
            t_fused(i) = t1 - t0
            sink = sink + chw_tensor(1, 1, 1)

            t0 = monotonic_now_ns()
            lb = preprocess_rgb_to_nchw_omp(img%view, img%width, img%height, INPUT_WIDTH, INPUT_HEIGHT, chw_tensor)
            t1 = monotonic_now_ns()
            t_omp(i) = t1 - t0
            sink = sink + chw_tensor(1, 1, 1)
        end do

        st_norm = compute_stats(t_norm, scratch)
        st_resize = compute_stats(t_resize, scratch)
        st_pad = compute_stats(t_pad, scratch)
        st_layout = compute_stats(t_layout, scratch)
        st_fused = compute_stats(t_fused, scratch)
        st_omp = compute_stats(t_omp, scratch)

        if (sink < -1.0_c_float) write(*, *) sink

        write(*, '(A)') ""
        write(*, '(A,A," (",I0,"x",I0," -> ",I0,"x",I0,"), ",I0," warmup + ",I0," measured")') &
            "Fortran Numerical Kernel Benchmark: ", trim(image_path), img%width, img%height, &
            lb%resized_w, lb%resized_h, warmup_iters, measured_iters
        write(*, '(A)') "---------------------------------------------------------------------"
        write(*, '(A)') "Kernel                 |    Mean |  Median |     P95 |     P99 |     Min |     Max |     FPS"
        call print_stats_row("1. RGB u8->f32 Norm", st_norm)
        call print_stats_row("2. Bilinear Resize", st_resize)
        call print_stats_row("3. Letterbox Pad", st_pad)
        call print_stats_row("4. Layout HWC->CHW", st_layout)
        call print_stats_row("5. Fused Preprocess", st_fused)
        call print_stats_row("6. Fused OpenMP", st_omp)
        write(*, '(A)') "====================================================================="
        write(*, '(A)') ""

        call image_destroy(img)
        deallocate(src_f32, resized_f32, padded_hwc, chw_tensor)
        deallocate(t_norm, t_resize, t_pad, t_layout, t_fused, t_omp, scratch)
    end subroutine run_kernel_benchmark

end module benchmark_mod
