//do this in build dir

export CFLAGS="-std=c11 -export-dynamic"
export CXXFLAGS="-std=c++14 -Wl -export-dynamic -fPIC"
cmake -DCMAKE_TOOLCHAIN_FILE= <Android SDK path>sdk/ndk/26.0.10792818/build/cmake/android.toolchain.cmake \
    -DTENSORFLOW_BUILD_DIR=<tflite_build path>tflite_build\
    -DTENSORFLOW_SOURCE_DIR=<tenserflow path>tensorflow \
    -DCMAKE_CROSSCOMPILING=true \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM="26" ..