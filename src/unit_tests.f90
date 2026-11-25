program fortran_unit_tests
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    use preprocessing_mod
    use postprocessing_mod
    use metrics_mod
    implicit none

    call test_compute_letterbox()
    call test_preprocess_rgb_to_nchw()
    call test_infer_and_decode()
    call test_compute_stats()
    write(*, '(A)') "All Fortran unit tests passed (4/4)"

contains

    subroutine expect(cond, msg)
        logical, intent(in) :: cond
        character(len=*), intent(in) :: msg
        if (.not. cond) then
            write(*, '(A,A)') "Assertion failed: ", msg
            stop 1
        end if
    end subroutine expect

    subroutine expect_approx(actual, expected, tol, msg)
        real(real64), intent(in) :: actual, expected, tol
        character(len=*), intent(in) :: msg
        if (abs(actual - expected) > tol) then
            write(*, '(A,A,A,F12.6,A,F12.6)') "Assertion failed (", msg, "): actual=", actual, " expected=", expected
            stop 1
        end if
    end subroutine expect_approx

    subroutine test_compute_letterbox()
        type(letterbox_params_t) :: p1, p2
        p1 = compute_letterbox(810_int32, 1080_int32, 640_int32, 640_int32)
        call expect(p1%resized_w == 480, "p1 resized_w")
        call expect(p1%resized_h == 640, "p1 resized_h")
        call expect(p1%pad_x == 80, "p1 pad_x")
        call expect(p1%pad_y == 0, "p1 pad_y")

        p2 = compute_letterbox(1280_int32, 720_int32, 640_int32, 640_int32)
        call expect(p2%resized_w == 640, "p2 resized_w")
        call expect(p2%resized_h == 360, "p2 resized_h")
        call expect(p2%pad_x == 0, "p2 pad_x")
        call expect(p2%pad_y == 140, "p2 pad_y")
    end subroutine test_compute_letterbox

    subroutine test_preprocess_rgb_to_nchw()
        integer(c_int8_t) :: src(3, 4, 2)
        real(c_float) :: out_tensor(4, 4, 3)
        type(letterbox_params_t) :: lb

        src = -1_c_int8_t ! 255 unsigned
        src(1, 1, 1) = -1_c_int8_t ! R=255
        src(2, 1, 1) = 0_c_int8_t  ! G=0
        src(3, 1, 1) = 0_c_int8_t  ! B=0

        lb = preprocess_rgb_to_nchw(src, 4_int32, 2_int32, 4_int32, 4_int32, out_tensor)
        call expect(lb%pad_y == 1, "pad_y == 1")
        call expect_approx(real(out_tensor(1, 1, 1), real64), real(PAD_VALUE_F32, real64), 1.0e-6_real64, "padding")
        call expect_approx(real(out_tensor(1, 2, 1), real64), 1.0_real64, 1.0e-5_real64, "R=1.0")
        call expect_approx(real(out_tensor(1, 2, 2), real64), 0.0_real64, 1.0e-5_real64, "G=0.0")
        call expect_approx(real(out_tensor(1, 2, 3), real64), 0.0_real64, 1.0e-5_real64, "B=0.0")
    end subroutine test_preprocess_rgb_to_nchw

    subroutine test_infer_and_decode()
        type(output_descriptor_t) :: d_e2e, d_raw, desc
        type(letterbox_params_t) :: lb
        type(scratch_buffer_t) :: scratch
        type(detection_t) :: out_dets(16)
        real(c_float) :: raw(6, 2)
        integer(int32) :: n
        logical :: ok

        ok = infer_output_descriptor([1_c_int64_t, 300_c_int64_t, 6_c_int64_t], d_e2e)
        call expect(ok .and. d_e2e%format == FMT_YOLO26_END2END, "end2end descriptor")
        call expect(.not. format_requires_nms(d_e2e%format), "end2end no NMS")

        ok = infer_output_descriptor([1_c_int64_t, 84_c_int64_t, 8400_c_int64_t], d_raw)
        call expect(ok .and. d_raw%format == FMT_RAW_CH_FIRST, "raw ch first")
        call expect(format_requires_nms(d_raw%format), "raw requires NMS")

        lb = compute_letterbox(810_int32, 1080_int32, 640_int32, 640_int32)
        raw(:, 1) = [80.0_c_float, 135.0_c_float, 560.0_c_float, 445.0_c_float, 0.92_c_float, 5.0_c_float]
        raw(:, 2) = [10.0_c_float, 10.0_c_float,  50.0_c_float,  50.0_c_float,  0.10_c_float, 0.0_c_float]
        ok = infer_output_descriptor([1_c_int64_t, 2_c_int64_t, 6_c_int64_t], desc)
        call expect(ok, "desc ok")

        call init_scratch(scratch, 16_int32)
        n = decode_output(raw, desc, lb, 0.25_real32, 0.45_real32, .false., scratch, out_dets)
        call expect(n == 1, "1 detection")
        call expect(out_dets(1)%class_id == 5, "bus class_id")
        call expect(out_dets(1)%x1 >= 0.0_real32 .and. out_dets(1)%x2 <= 810.0_real32, "x bounds")
        call expect(out_dets(1)%y1 >= 0.0_real32 .and. out_dets(1)%y2 <= 1080.0_real32, "y bounds")
        call destroy_scratch(scratch)
    end subroutine test_infer_and_decode

    subroutine test_compute_stats()
        integer(int64) :: samples(100), scratch(100)
        type(latency_stats_t) :: st
        integer :: i

        do i = 1, 100
            samples(i) = int(i, int64) * 1000000_int64
        end do
        st = compute_stats(samples, scratch)
        call expect_approx(st%min_ms, 1.0_real64, 1.0e-6_real64, "min_ms")
        call expect_approx(st%max_ms, 100.0_real64, 1.0e-6_real64, "max_ms")
        call expect_approx(st%mean_ms, 50.5_real64, 1.0e-6_real64, "mean_ms")
        call expect_approx(st%median_ms, 50.5_real64, 1.0e-6_real64, "median_ms")
        call expect_approx(st%p95_ms, 95.05_real64, 1.0e-6_real64, "p95_ms")
        call expect_approx(st%p99_ms, 99.01_real64, 1.0e-6_real64, "p99_ms")
    end subroutine test_compute_stats

end program fortran_unit_tests
