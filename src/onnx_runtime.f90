module onnx_runtime_mod
    use, intrinsic :: iso_c_binding
    implicit none
    private

    public :: ort_session_t
    public :: ort_init_session
    public :: ort_bind_tensors
    public :: ort_run_inference
    public :: ort_destroy_session

    ! Memory ownership contract:
    ! - `ort_session_t%handle` owns the C-side OrtEnv, OrtSessionOptions, OrtSession,
    !   OrtMemoryInfo, and OrtValue wrappers.
    ! - Fortran TARGET arrays `input_buf` and `output_buf` passed to `ort_bind_tensors`
    !   remain owned by Fortran (`detector_t`). No tensor data is copied by ORT.
    ! - `ort_destroy_session` MUST be called before deallocating the Fortran TARGET buffers.
    type :: ort_session_t
        type(c_ptr) :: handle = c_null_ptr
        character(len=64) :: ort_version = ""
        integer(c_int64_t) :: input_dims(4) = 0
        integer(c_int64_t) :: output_dims(3) = 0
    end type ort_session_t

    interface
        function ort_bridge_create(model_path, intra_threads, inter_threads, &
                out_ver, ver_cap, out_in_dims, out_out_dims, err_buf, err_cap) &
                bind(c, name="ort_bridge_create")
            import :: c_ptr, c_char, c_int32_t, c_int64_t, c_size_t
            character(kind=c_char), intent(in) :: model_path(*)
            integer(c_int32_t), value :: intra_threads
            integer(c_int32_t), value :: inter_threads
            character(kind=c_char), intent(out) :: out_ver(*)
            integer(c_size_t), value :: ver_cap
            integer(c_int64_t), intent(out) :: out_in_dims(4)
            integer(c_int64_t), intent(out) :: out_out_dims(3)
            character(kind=c_char), intent(out) :: err_buf(*)
            integer(c_size_t), value :: err_cap
            type(c_ptr) :: ort_bridge_create
        end function ort_bridge_create

        function ort_bridge_bind_tensors(ctx, in_ptr, in_bytes, in_dims, &
                out_ptr, out_bytes, out_dims, err_buf, err_cap) &
                bind(c, name="ort_bridge_bind_tensors")
            import :: c_ptr, c_int, c_size_t, c_int64_t, c_char
            type(c_ptr), value :: ctx
            type(c_ptr), value :: in_ptr
            integer(c_size_t), value :: in_bytes
            integer(c_int64_t), intent(in) :: in_dims(4)
            type(c_ptr), value :: out_ptr
            integer(c_size_t), value :: out_bytes
            integer(c_int64_t), intent(in) :: out_dims(3)
            character(kind=c_char), intent(out) :: err_buf(*)
            integer(c_size_t), value :: err_cap
            integer(c_int) :: ort_bridge_bind_tensors
        end function ort_bridge_bind_tensors

        function ort_bridge_run(ctx, err_buf, err_cap) bind(c, name="ort_bridge_run")
            import :: c_ptr, c_int, c_char, c_size_t
            type(c_ptr), value :: ctx
            character(kind=c_char), intent(out) :: err_buf(*)
            integer(c_size_t), value :: err_cap
            integer(c_int) :: ort_bridge_run
        end function ort_bridge_run

        subroutine ort_bridge_destroy(ctx) bind(c, name="ort_bridge_destroy")
            import :: c_ptr
            type(c_ptr), value :: ctx
        end subroutine ort_bridge_destroy
    end interface

contains

    subroutine copy_c_str(c_arr, f_str)
        character(kind=c_char), intent(in) :: c_arr(:)
        character(len=*), intent(out) :: f_str
        integer :: i, n

        f_str = ""
        n = min(size(c_arr), len(f_str))
        do i = 1, n
            if (c_arr(i) == c_null_char) exit
            f_str(i:i) = c_arr(i)
        end do
    end subroutine copy_c_str

    function to_c_str(f_str) result(c_arr)
        character(len=*), intent(in) :: f_str
        character(kind=c_char), allocatable :: c_arr(:)
        integer :: i, n

        n = len_trim(f_str)
        allocate(c_arr(n + 1))
        do i = 1, n
            c_arr(i) = f_str(i:i)
        end do
        c_arr(n + 1) = c_null_char
    end function to_c_str

    function ort_init_session(session, model_path, intra_threads, inter_threads, err_msg) result(ok)
        type(ort_session_t), intent(out) :: session
        character(len=*), intent(in) :: model_path
        integer(c_int32_t), intent(in) :: intra_threads, inter_threads
        character(len=*), intent(out) :: err_msg
        logical :: ok

        character(kind=c_char), allocatable :: c_path(:)
        character(kind=c_char) :: c_ver(64), c_err(512)

        err_msg = ""
        c_ver = c_null_char
        c_err = c_null_char
        c_path = to_c_str(model_path)

        session%handle = ort_bridge_create( &
            c_path, intra_threads, inter_threads, &
            c_ver, int(size(c_ver), c_size_t), &
            session%input_dims, session%output_dims, &
            c_err, int(size(c_err), c_size_t))

        deallocate(c_path)
        call copy_c_str(c_ver, session%ort_version)

        if (.not. c_associated(session%handle)) then
            call copy_c_str(c_err, err_msg)
            ok = .false.
        else
            ok = .true.
        end if
    end function ort_init_session

    function ort_bind_tensors(session, input_buf, output_buf, err_msg) result(ok)
        type(ort_session_t), intent(inout) :: session
        real(c_float), target, intent(inout) :: input_buf(:, :, :)
        real(c_float), target, intent(inout) :: output_buf(:, :)
        character(len=*), intent(out) :: err_msg
        logical :: ok

        character(kind=c_char) :: c_err(512)
        integer(c_size_t) :: in_bytes, out_bytes
        integer(c_int) :: rc

        err_msg = ""
        c_err = c_null_char
        in_bytes = int(size(input_buf), c_size_t) * 4_c_size_t
        out_bytes = int(size(output_buf), c_size_t) * 4_c_size_t

        rc = ort_bridge_bind_tensors( &
            session%handle, &
            c_loc(input_buf(1, 1, 1)), in_bytes, session%input_dims, &
            c_loc(output_buf(1, 1)), out_bytes, session%output_dims, &
            c_err, int(size(c_err), c_size_t))

        if (rc /= 0) then
            call copy_c_str(c_err, err_msg)
            ok = .false.
        else
            ok = .true.
        end if
    end function ort_bind_tensors

    function ort_run_inference(session, err_msg) result(ok)
        type(ort_session_t), intent(inout) :: session
        character(len=*), intent(out) :: err_msg
        logical :: ok

        character(kind=c_char) :: c_err(512)
        integer(c_int) :: rc

        err_msg = ""
        c_err = c_null_char
        rc = ort_bridge_run(session%handle, c_err, int(size(c_err), c_size_t))
        if (rc /= 0) then
            call copy_c_str(c_err, err_msg)
            ok = .false.
        else
            ok = .true.
        end if
    end function ort_run_inference

    subroutine ort_destroy_session(session)
        type(ort_session_t), intent(inout) :: session
        if (c_associated(session%handle)) then
            call ort_bridge_destroy(session%handle)
            session%handle = c_null_ptr
        end if
    end subroutine ort_destroy_session

end module onnx_runtime_mod
