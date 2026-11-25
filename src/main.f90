program fortran_yolo26n
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use image_mod
    use preprocessing_mod
    use postprocessing_mod
    use visualization_mod
    use detector_mod
    use metrics_mod
    use benchmark_mod
    implicit none

    integer(int64) :: app_start_ns
    character(len=32) :: mode
    character(len=512) :: model_path, output_image, csv_out, json_out
    character(len=512) :: image_paths(16)
    integer :: image_count, warmup_iters, measured_iters
    integer(int32) :: threads
    real(real32) :: conf_threshold, iou_threshold
    logical :: force_nms

    app_start_ns = monotonic_now_ns()

    mode = "detect"
    model_path = "models/yolov26n.onnx"
    output_image = ""
    csv_out = "benchmarks/fortran_benchmark_results.csv"
    json_out = "benchmarks/fortran_benchmark_results.json"
    image_paths = ""
    image_count = 0
    threads = 4
    warmup_iters = 100
    measured_iters = 1000
    conf_threshold = 0.25_real32
    iou_threshold = 0.45_real32
    force_nms = .false.

    call parse_cli()

    if (image_count == 0) then
        image_paths(1) = "test/bus.jpg"
        image_paths(2) = "test/zidane.jpg"
        image_count = 2
    end if

    select case (trim(mode))
    case ("detect")
        call run_detect(image_paths(1))
    case ("validate")
        call run_validate()
    case ("benchmark")
        call run_bench_mode()
    case ("kernel-bench")
        call run_kernel_benchmark(image_paths(1), warmup_iters, measured_iters)
    case default
        write(*, '(A,A)') "Unknown mode: ", trim(mode)
        stop 1
    end select

contains

    subroutine print_usage()
        write(*, '(A)') "Usage: fortran-yolo26n [detect|validate|benchmark|kernel-bench] [options]"
        write(*, '(A)') ""
        write(*, '(A)') "Options:"
        write(*, '(A)') "  --model <path>     ONNX model path (default: models/yolov26n.onnx)"
        write(*, '(A)') "  --image <path>     Input image path (repeatable; defaults to test/bus.jpg, test/zidane.jpg)"
        write(*, '(A)') "  --output <path>    Save annotated PNG image in detect mode"
        write(*, '(A)') "  --threads <n>      ONNX Runtime intra-op CPU threads (default: 4)"
        write(*, '(A)') "  --warmup <n>       Warmup iterations (default: 100)"
        write(*, '(A)') "  --iters <n>        Measured iterations (default: 1000)"
        write(*, '(A)') "  --conf <f32>       Confidence threshold (default: 0.25)"
        write(*, '(A)') "  --iou <f32>        IoU threshold (default: 0.45)"
        write(*, '(A)') "  --force-nms        Force NMS even on end2end [1,N,6] outputs"
        write(*, '(A)') "  --csv <path>       Benchmark CSV output path"
        write(*, '(A)') "  --json <path>      Benchmark JSON output path"
        write(*, '(A)') "  -h, --help         Show this help message"
    end subroutine print_usage

    subroutine parse_cli()
        integer :: argc, i, ios
        character(len=512) :: arg, val

        argc = command_argument_count()
        i = 1
        do while (i <= argc)
            call get_command_argument(i, arg)
            select case (trim(arg))
            case ("detect", "validate", "benchmark", "kernel-bench")
                mode = trim(arg)
            case ("-h", "--help")
                call print_usage()
                stop 0
            case ("--model")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, model_path)
            case ("--image")
                i = i + 1; if (i > argc) stop 1
                if (image_count >= 16) stop 1
                image_count = image_count + 1
                call get_command_argument(i, image_paths(image_count))
            case ("--output")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, output_image)
            case ("--csv")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, csv_out)
            case ("--json")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, json_out)
            case ("--threads")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, val)
                read(val, *, iostat=ios) threads
                if (ios /= 0) stop 1
            case ("--warmup")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, val)
                read(val, *, iostat=ios) warmup_iters
                if (ios /= 0) stop 1
            case ("--iters")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, val)
                read(val, *, iostat=ios) measured_iters
                if (ios /= 0) stop 1
            case ("--conf")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, val)
                read(val, *, iostat=ios) conf_threshold
                if (ios /= 0) stop 1
            case ("--iou")
                i = i + 1; if (i > argc) stop 1
                call get_command_argument(i, val)
                read(val, *, iostat=ios) iou_threshold
                if (ios /= 0) stop 1
            case ("--force-nms")
                force_nms = .true.
            case default
                write(*, '(A,A)') "Unknown argument: ", trim(arg)
                stop 1
            end select
            i = i + 1
        end do
    end subroutine parse_cli

    subroutine run_detect(img_path)
        character(len=*), intent(in) :: img_path
        type(detector_t) :: det
        type(detector_config_t) :: dcfg
        type(image_t) :: img, vis
        type(letterbox_params_t) :: lb
        character(len=512) :: err_msg
        character(len=:), allocatable :: fmt_name, cname
        integer(int64) :: t0, t1, t2, t3
        integer(int32) :: det_count, i

        dcfg = detector_config_t(model_path, threads, 1_int32, conf_threshold, iou_threshold, force_nms)
        if (.not. detector_init(det, dcfg, err_msg)) then
            write(*, '(A,A)') "Error: ", trim(err_msg)
            stop 1
        end if

        if (.not. image_load(img_path, img)) then
            write(*, '(A,A)') "Failed to load image: ", trim(img_path)
            call detector_destroy(det)
            stop 1
        end if

        t0 = monotonic_now_ns()
        lb = detector_preprocess(det, img%view, img%width, img%height)
        t1 = monotonic_now_ns()
        if (.not. detector_infer_raw(det, err_msg)) then
            write(*, '(A,A)') "Inference error: ", trim(err_msg)
            call image_destroy(img)
            call detector_destroy(det)
            stop 1
        end if
        t2 = monotonic_now_ns()
        det_count = detector_postprocess(det, lb)
        t3 = monotonic_now_ns()

        fmt_name = output_format_name(det%output_desc%format)
        write(*, '(A,A," (",I0,"x",I0,"), Output format: ",A)') "Image: ", trim(img_path), img%width, img%height, fmt_name
        write(*, '("Latency: pre=",F0.3,"ms, infer=",F0.3,"ms, post=",F0.3,"ms, total=",F0.3,"ms")') &
            real(t1 - t0, real64) / 1.0e6_real64, &
            real(t2 - t1, real64) / 1.0e6_real64, &
            real(t3 - t2, real64) / 1.0e6_real64, &
            real(t3 - t0, real64) / 1.0e6_real64
        write(*, '("Detections (",I0,"):")') det_count
        do i = 1, det_count
            cname = coco_class_name(det%detections(i)%class_id)
            write(*, '("  [",I0,"] class=",I0," (",A,") conf=",F0.4," box=[",F0.2,", ",F0.2,", ",F0.2,", ",F0.2,"]")') &
                i - 1, det%detections(i)%class_id, cname, det%detections(i)%confidence, &
                det%detections(i)%x1, det%detections(i)%y1, det%detections(i)%x2, det%detections(i)%y2
        end do

        if (len_trim(output_image) > 0) then
            call image_clone(img, vis)
            call draw_detections(vis, det%detections, det_count)
            if (.not. image_save_png(vis, output_image)) then
                write(*, '(A,A)') "Failed to save PNG: ", trim(output_image)
                call image_destroy(vis)
                call image_destroy(img)
                call detector_destroy(det)
                stop 1
            end if
            call image_destroy(vis)
        end if

        call image_destroy(img)
        call detector_destroy(det)
    end subroutine run_detect

    subroutine run_validate()
        type(detector_t) :: det
        type(detector_config_t) :: dcfg
        type(image_t) :: img
        type(letterbox_params_t) :: lb
        type(process_memory_t) :: mem0, mem1
        character(len=512) :: err_msg
        character(len=:), allocatable :: fmt_name
        integer(int32) :: det_count, i, j
        real(real32) :: fw, fh

        dcfg = detector_config_t(model_path, threads, 1_int32, conf_threshold, iou_threshold, force_nms)
        if (.not. detector_init(det, dcfg, err_msg)) then
            write(*, '(A,A)') "Validation failed: ", trim(err_msg)
            stop 1
        end if

        fmt_name = output_format_name(det%output_desc%format)
        write(*, '("[PASS] Model loaded: ",A," (ORT ",A,")")') trim(model_path), trim(det%session%ort_version)
        write(*, '("[PASS] Input tensor validated: [",I0,", ",I0,", ",I0,", ",I0,"] FP32")') &
            det%session%input_dims(1), det%session%input_dims(2), det%session%input_dims(3), det%session%input_dims(4)
        write(*, '("[PASS] Output tensor validated: [",I0,", ",I0,", ",I0,"] (",A,")")') &
            det%session%output_dims(1), det%session%output_dims(2), det%session%output_dims(3), fmt_name

        do i = 1, image_count
            if (.not. image_load(image_paths(i), img)) then
                write(*, '(A,A)') "Failed to load image: ", trim(image_paths(i))
                stop 1
            end if
            lb = detector_preprocess(det, img%view, img%width, img%height)
            if (.not. detector_infer_raw(det, err_msg)) then
                write(*, '(A,A)') "Inference failed: ", trim(err_msg)
                stop 1
            end if
            det_count = detector_postprocess(det, lb)
            if (det_count <= 0) then
                write(*, '(A,A)') "No detections found on ", trim(image_paths(i))
                stop 1
            end if

            fw = real(img%width, real32)
            fh = real(img%height, real32)
            do j = 1, det_count
                if (det%detections(j)%x1 < 0.0_real32 .or. det%detections(j)%y1 < 0.0_real32 .or. &
                    det%detections(j)%x2 > fw .or. det%detections(j)%y2 > fh .or. &
                    det%detections(j)%x2 <= det%detections(j)%x1 .or. det%detections(j)%y2 <= det%detections(j)%y1) then
                    write(*, '(A)') "Bounding box out of bounds"
                    stop 1
                end if
                if (.not. ieee_is_finite(det%detections(j)%confidence) .or. &
                    det%detections(j)%confidence < conf_threshold .or. det%detections(j)%confidence > 1.0_real32) then
                    write(*, '(A)') "Invalid confidence value"
                    stop 1
                end if
                if (det%detections(j)%class_id < 0 .or. det%detections(j)%class_id >= NUM_COCO_CLASSES) then
                    write(*, '(A)') "Invalid class_id"
                    stop 1
                end if
            end do
            write(*, '("[PASS] ",A,": ",I0," valid detections, all boxes within [0..",I0,", 0..",I0,"]")') &
                trim(image_paths(i)), det_count, img%width, img%height
            call image_destroy(img)
        end do

        mem0 = read_process_memory()
        do i = 1, 50
            if (.not. detector_infer_raw(det, err_msg)) stop 1
        end do
        mem1 = read_process_memory()
        write(*, '("[PASS] 50 repeated inferences RSS check: ",I0," KiB -> ",I0," KiB")') &
            mem0%current_rss_kb, mem1%current_rss_kb

        call detector_destroy(det)
    end subroutine run_validate

    subroutine run_bench_mode()
        type(benchmark_config_t) :: bcfg
        type(benchmark_report_t) :: rep

        bcfg%model_path = model_path
        bcfg%image_paths = image_paths
        bcfg%image_count = image_count
        bcfg%warmup_iters = warmup_iters
        bcfg%measured_iters = measured_iters
        bcfg%threads = threads
        bcfg%conf_threshold = conf_threshold
        bcfg%iou_threshold = iou_threshold
        bcfg%force_nms = force_nms
        bcfg%csv_out_path = csv_out
        bcfg%json_out_path = json_out

        rep = run_benchmark(app_start_ns, bcfg)
    end subroutine run_bench_mode

end program fortran_yolo26n
