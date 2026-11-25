module image_mod
    use, intrinsic :: iso_c_binding
    implicit none
    private

    public :: image_t
    public :: image_load
    public :: image_clone
    public :: image_save_png
    public :: image_destroy

    ! Stores RGB24 image in column-major (3, width, height) matching C row-major
    ! (height, width, 3) memory layout with zero transpose or copy.
    type :: image_t
        integer(c_int32_t) :: width = 0
        integer(c_int32_t) :: height = 0
        integer(c_int32_t) :: channels = 3
        type(c_ptr) :: c_buf = c_null_ptr
        integer(c_int8_t), pointer :: view(:, :, :) => null()
        integer(c_int8_t), allocatable :: owned(:, :, :)
        logical :: owns_stb = .false.
    end type image_t

    interface
        function img_bridge_load_rgb(path, out_w, out_h) bind(c, name="img_bridge_load_rgb")
            import :: c_ptr, c_char, c_int32_t
            character(kind=c_char), intent(in) :: path(*)
            integer(c_int32_t), intent(out) :: out_w
            integer(c_int32_t), intent(out) :: out_h
            type(c_ptr) :: img_bridge_load_rgb
        end function img_bridge_load_rgb

        subroutine img_bridge_free(ptr) bind(c, name="img_bridge_free")
            import :: c_ptr
            type(c_ptr), value :: ptr
        end subroutine img_bridge_free

        function img_bridge_save_png(path, w, h, rgb_data) bind(c, name="img_bridge_save_png")
            import :: c_char, c_int32_t, c_ptr
            character(kind=c_char), intent(in) :: path(*)
            integer(c_int32_t), value :: w
            integer(c_int32_t), value :: h
            type(c_ptr), value :: rgb_data
            integer(c_int32_t) :: img_bridge_save_png
        end function img_bridge_save_png
    end interface

contains

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

    function image_load(path, img) result(ok)
        character(len=*), intent(in) :: path
        type(image_t), intent(out) :: img
        logical :: ok

        character(kind=c_char), allocatable :: c_path(:)
        integer(c_int32_t) :: w, h

        c_path = to_c_str(path)
        img%c_buf = img_bridge_load_rgb(c_path, w, h)
        deallocate(c_path)

        if (.not. c_associated(img%c_buf) .or. w <= 0 .or. h <= 0) then
            ok = .false.
            return
        end if

        img%width = w
        img%height = h
        img%channels = 3
        img%owns_stb = .true.
        call c_f_pointer(img%c_buf, img%view, [3_c_size_t, int(w, c_size_t), int(h, c_size_t)])
        ok = .true.
    end function image_load

    subroutine image_clone(src, dst)
        type(image_t), intent(in) :: src
        type(image_t), intent(out), target :: dst

        dst%width = src%width
        dst%height = src%height
        dst%channels = src%channels
        dst%owns_stb = .false.
        allocate(dst%owned(3, src%width, src%height))
        dst%owned(:, :, :) = src%view(:, :, :)
        dst%view => dst%owned
    end subroutine image_clone

    function image_save_png(img, path) result(ok)
        type(image_t), intent(in), target :: img
        character(len=*), intent(in) :: path
        logical :: ok

        character(kind=c_char), allocatable :: c_path(:)
        type(c_ptr) :: ptr
        integer(c_int32_t) :: rc

        if (img%owns_stb) then
            ptr = img%c_buf
        else if (allocated(img%owned)) then
            ptr = c_loc(img%owned(1, 1, 1))
        else
            ok = .false.
            return
        end if

        c_path = to_c_str(path)
        rc = img_bridge_save_png(c_path, img%width, img%height, ptr)
        deallocate(c_path)
        ok = (rc /= 0)
    end function image_save_png

    subroutine image_destroy(img)
        type(image_t), intent(inout) :: img

        nullify(img%view)
        if (img%owns_stb .and. c_associated(img%c_buf)) then
            call img_bridge_free(img%c_buf)
            img%c_buf = c_null_ptr
            img%owns_stb = .false.
        end if
        if (allocated(img%owned)) then
            deallocate(img%owned)
        end if
        img%width = 0
        img%height = 0
    end subroutine image_destroy

end module image_mod
