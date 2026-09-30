cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(mlbackend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------
# ONNX Runtime (mandatory): the onnxruntime-cpp conda-forge package installs
# its C API header/lib under $CONDA_PREFIX/include, $CONDA_PREFIX/lib, which
# CMAKE_PREFIX_PATH=$CONDA_PREFIX (the environment.yml-created env is active
# by default) makes find_path/find_library see. Configure fails without it.
# ------------------------------------------------------------------------
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime/core/session/onnxruntime_c_api.h)
find_library(ONNXRUNTIME_LIBRARY onnxruntime)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "onnxruntime-cpp not found (need both onnxruntime_c_api.h and "
        "libonnxruntime). Activate an environment with onnxruntime-cpp "
        "installed (see ../environment.yml) or pass -DCMAKE_PREFIX_PATH "
        "to one.")
endif()

# ------------------------------------------------------------------------
# LibTorch (mandatory): the pytorch-cpu conda-forge package ships it, and
# CMAKE_PREFIX_PATH=torch.utils.cmake_prefix_path (set by the Makefile)
# makes find_package(Torch) see it. Configure fails without it.
# ------------------------------------------------------------------------
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(FATAL_ERROR
        "LibTorch not found. Activate an environment with PyTorch installed "
        "(see ../environment.yml) or set -DCMAKE_PREFIX_PATH to "
        "\"$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')\".")
endif()

add_library(mlbackend SHARED ml_backend.cpp ml_backend.h)

target_include_directories(mlbackend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(mlbackend PUBLIC ${ONNXRUNTIME_LIBRARY})

target_link_libraries(mlbackend PUBLIC ${TORCH_LIBRARIES})
target_compile_options(mlbackend PRIVATE ${TORCH_CXX_FLAGS})

# Installed into the active environment (e.g. $CONDA_PREFIX), so
# Source_ML.comp's `DEPENDENCY "-lmlbackend"` line -- which relies on
# mcstas's own conda-aware CFLAGS for -I/-L/-rpath, not this build's own
# paths -- can find it.
install(TARGETS mlbackend LIBRARY DESTINATION lib RUNTIME DESTINATION lib)
install(FILES ml_backend.h DESTINATION include)

# Pure-C smoke test: proves the library is callable from C alone (what
# mcrun's generated instrument code will actually do), for both an .onnx and
# (when built) a .pt/.pth model.
enable_language(C)
add_executable(c_smoke_test c_smoke_test.c)
target_link_libraries(c_smoke_test PRIVATE mlbackend)
