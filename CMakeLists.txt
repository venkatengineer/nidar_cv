cmake_minimum_required(VERSION 3.16)
project(fortran-yolo26n LANGUAGES C Fortran)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

option(ENABLE_VECTOR_REPORT "Emit GFortran vectorization report (-fopt-info-vec-optimized)" OFF)

find_package(OpenMP REQUIRED COMPONENTS Fortran)

set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Path to ONNX Runtime headers")
set(ORT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib" CACHE PATH "Path to libonnxruntime.so")

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ORT_LIB_DIR}" /usr/lib /usr/local/lib
    REQUIRED
)

add_library(fortran_yolo_core STATIC
    c/onnx_bridge.c
    c/image_bridge.c
    c/system_metrics.c
    src/onnx_runtime.f90
    src/image.f90
    src/preprocessing.f90
    src/postprocessing.f90
    src/visualization.f90
    src/detector.f90
    src/metrics.f90
    src/benchmark.f90
)

target_include_directories(fortran_yolo_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ORT_INCLUDE_DIR}"
)

target_link_libraries(fortran_yolo_core PUBLIC
    "${ONNXRUNTIME_LIB}"
    OpenMP::OpenMP_Fortran
    m
)

target_compile_options(fortran_yolo_core PUBLIC
    $<$<COMPILE_LANGUAGE:Fortran>:-cpp -std=f2018 -Wall -Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-g -O0 -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow>
    $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-O3 -march=native -mtune=native -DNDEBUG>
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:-O3 -march=native -mtune=native -DNDEBUG>
    $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<BOOL:${ENABLE_VECTOR_REPORT}>>:-fopt-info-vec-optimized>
)

add_executable(fortran-yolo26n src/main.f90)
target_link_libraries(fortran-yolo26n PRIVATE fortran_yolo_core)
set_target_properties(fortran-yolo26n PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};/usr/lib"
    INSTALL_RPATH "$ORIGIN/../../lib;$ORIGIN/../lib;/usr/lib"
)

enable_testing()
add_executable(fortran-yolo26n-tests src/unit_tests.f90)
target_link_libraries(fortran-yolo26n-tests PRIVATE fortran_yolo_core)
add_test(NAME fortran_unit_tests COMMAND fortran-yolo26n-tests)
