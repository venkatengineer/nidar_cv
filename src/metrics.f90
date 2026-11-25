module metrics_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private

    public :: latency_stats_t, process_memory_t
    public :: monotonic_now_ns, process_cpu_now_ns
    public :: read_process_memory, read_executable_size_bytes
    public :: compute_stats, compute_cpu_utilization_pct

    type :: latency_stats_t
        integer :: count = 0
        real(real64) :: min_ms = 0.0_real64
        real(real64) :: max_ms = 0.0_real64
        real(real64) :: mean_ms = 0.0_real64
        real(real64) :: std_dev_ms = 0.0_real64
        real(real64) :: median_ms = 0.0_real64
        real(real64) :: p95_ms = 0.0_real64
        real(real64) :: p99_ms = 0.0_real64
        real(real64) :: avg_fps = 0.0_real64
    end type latency_stats_t

    type :: process_memory_t
        integer(int64) :: peak_rss_kb = 0
        integer(int64) :: current_rss_kb = 0
    end type process_memory_t

    interface
        function sys_monotonic_now_ns() bind(c, name="sys_monotonic_now_ns")
            import :: c_int64_t
            integer(c_int64_t) :: sys_monotonic_now_ns
        end function sys_monotonic_now_ns

        function sys_process_cpu_now_ns() bind(c, name="sys_process_cpu_now_ns")
            import :: c_int64_t
            integer(c_int64_t) :: sys_process_cpu_now_ns
        end function sys_process_cpu_now_ns

        subroutine sys_read_process_memory(peak_kb, curr_kb) bind(c, name="sys_read_process_memory")
            import :: c_int64_t
            integer(c_int64_t), intent(out) :: peak_kb
            integer(c_int64_t), intent(out) :: curr_kb
        end subroutine sys_read_process_memory

        function sys_executable_size_bytes() bind(c, name="sys_executable_size_bytes")
            import :: c_int64_t
            integer(c_int64_t) :: sys_executable_size_bytes
        end function sys_executable_size_bytes
    end interface

contains

    function monotonic_now_ns() result(ns)
        integer(int64) :: ns
        ns = sys_monotonic_now_ns()
    end function monotonic_now_ns

    function process_cpu_now_ns() result(ns)
        integer(int64) :: ns
        ns = sys_process_cpu_now_ns()
    end function process_cpu_now_ns

    function read_process_memory() result(mem)
        type(process_memory_t) :: mem
        integer(c_int64_t) :: p, c
        call sys_read_process_memory(p, c)
        mem%peak_rss_kb = p
        mem%current_rss_kb = c
    end function read_process_memory

    function read_executable_size_bytes() result(sz)
        integer(int64) :: sz
        sz = sys_executable_size_bytes()
    end function read_executable_size_bytes

    recursive subroutine quicksort_i64(a, left, right)
        integer(int64), intent(inout) :: a(:)
        integer, intent(in) :: left, right
        integer :: i, j
        integer(int64) :: pivot, tmp

        if (left >= right) return
        pivot = a((left + right) / 2)
        i = left
        j = right
        do
            do while (a(i) < pivot)
                i = i + 1
            end do
            do while (a(j) > pivot)
                j = j - 1
            end do
            if (i <= j) then
                tmp = a(i)
                a(i) = a(j)
                a(j) = tmp
                i = i + 1
                j = j - 1
            end if
            if (i > j) exit
        end do
        if (left < j) call quicksort_i64(a, left, j)
        if (i < right) call quicksort_i64(a, i, right)
    end subroutine quicksort_i64

    function percentile_ns(sorted_ns, p) result(val)
        integer(int64), intent(in) :: sorted_ns(:)
        real(real64), intent(in) :: p
        real(real64) :: val, pos, frac, v_lo, v_hi
        integer :: n, lo, hi

        n = size(sorted_ns)
        if (n == 1) then
            val = real(sorted_ns(1), real64)
            return
        end if
        pos = p * real(n - 1, real64)
        lo = int(floor(pos)) + 1
        hi = min(lo + 1, n)
        frac = pos - real(lo - 1, real64)
        v_lo = real(sorted_ns(lo), real64)
        v_hi = real(sorted_ns(hi), real64)
        val = v_lo + frac * (v_hi - v_lo)
    end function percentile_ns

    function compute_stats(samples_ns, scratch_sorted) result(st)
        integer(int64), intent(in) :: samples_ns(:)
        integer(int64), intent(inout) :: scratch_sorted(:)
        type(latency_stats_t) :: st

        integer :: i, n
        real(real64) :: sum_ns, mean_ns, var_sum, diff, std_dev_ns, mean_ms
        real(real64), parameter :: NS_TO_MS = 1.0_real64 / 1000000.0_real64

        n = size(samples_ns)
        scratch_sorted(1:n) = samples_ns(1:n)
        call quicksort_i64(scratch_sorted, 1, n)

        sum_ns = 0.0_real64
        do i = 1, n
            sum_ns = sum_ns + real(scratch_sorted(i), real64)
        end do
        mean_ns = sum_ns / real(n, real64)

        var_sum = 0.0_real64
        do i = 1, n
            diff = real(scratch_sorted(i), real64) - mean_ns
            var_sum = var_sum + diff * diff
        end do
        std_dev_ns = sqrt(var_sum / real(n, real64))
        mean_ms = mean_ns * NS_TO_MS

        st%count = n
        st%min_ms = real(scratch_sorted(1), real64) * NS_TO_MS
        st%max_ms = real(scratch_sorted(n), real64) * NS_TO_MS
        st%mean_ms = mean_ms
        st%std_dev_ms = std_dev_ns * NS_TO_MS
        st%median_ms = percentile_ns(scratch_sorted, 0.50_real64) * NS_TO_MS
        st%p95_ms = percentile_ns(scratch_sorted, 0.95_real64) * NS_TO_MS
        st%p99_ms = percentile_ns(scratch_sorted, 0.99_real64) * NS_TO_MS
        if (mean_ms > 0.0_real64) then
            st%avg_fps = 1000.0_real64 / mean_ms
        else
            st%avg_fps = 0.0_real64
        end if
    end function compute_stats

    pure function compute_cpu_utilization_pct(cpu_ns, wall_ns) result(pct)
        integer(int64), intent(in) :: cpu_ns, wall_ns
        real(real64) :: pct
        if (wall_ns <= 0 .or. cpu_ns < 0) then
            pct = 0.0_real64
        else
            pct = (real(cpu_ns, real64) / real(wall_ns, real64)) * 100.0_real64
        end if
    end function compute_cpu_utilization_pct

end module metrics_mod
