cmake_minimum_required(VERSION 3.8)
project(demo C CXX)
include_directories(
    "${TENSORFLOW_SOURCE_DIR}"
    "${TENSORFLOW_BUILD_DIR}/abseil-cpp/"
    "${TENSORFLOW_BUILD_DIR}/flatbuffers/include"
    )
find_library(
    ${TENSORFLOW_BUILD_DIR} libtensorflow-lite.a
)
set(DEMO_SOURCES "")
list(APPEND DEMO_SOURCES
    "main.cpp"
)
set(LIB_PATHS "")
list(APPEND LIB_PATHS
    "${TENSORFLOW_BUILD_DIR}"
    "${TENSORFLOW_BUILD_DIR}/_deps/xnnpack-build"
    "${TENSORFLOW_BUILD_DIR}/_deps/ruy-build/ruy"
    "${TENSORFLOW_BUILD_DIR}/_deps/abseil-cpp-build/absl/status"
    "${TENSORFLOW_BUILD_DIR}/_deps/abseil-cpp-build/absl/strings"
    "${TENSORFLOW_BUILD_DIR}/_deps/abseil-cpp-build/absl/flags"
    "${TENSORFLOW_BUILD_DIR}/_deps/farmhash-build"
    "${TENSORFLOW_BUILD_DIR}/pthreadpool"
    "${TENSORFLOW_BUILD_DIR}/_deps/flatbuffers-build"
    "${TENSORFLOW_BUILD_DIR}/_deps/cpuinfo-build"
)
set(LIBS "")
foreach(_path IN LISTS LIB_PATHS)
    message("Linking: ${_path}")
    link_directories(${_path})
    file(GLOB TEMP_LIB "${_path}/*.a")
    foreach(_lib_path IN LISTS TEMP_LIB)
        get_filename_component(_lib_name ${_lib_path} NAME)
        list(APPEND LIBS ${_lib_name})
    endforeach()
endforeach()
link_directories(
    "${TENSORFLOW_BUILD_DIR}/_deps/fft2d-build"
    "${TENSORFLOW_BUILD_DIR}/_deps/abseil-cpp-build/absl/status"
)
add_executable(demo ${DEMO_SOURCES})
target_link_libraries(demo
    ${LIBS}
    absl_flags
    #absl::flags
    #absl::flags_parse
    #absl::status
    android
    log
    libtensorflow-lite.a
    libfft2d_fftsg2d.a
    libfft2d_fftsg.a
)