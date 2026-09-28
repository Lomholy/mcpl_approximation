cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(mlbackend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------
# ONNX Runtime (mandatory): the onnxruntime-cpp conda-forge package installs
# its C API header/lib under $CONDA_PREFIX/include, $CONDA_PREFIX/lib, which
# CMAKE_PREFIX_PATH=$CONDA_PREFIX (the environment.yml-created env is active
# by default) makes find_path/find_library see. Unlike LibTorch, this is not
# optional: onnxruntime-cpp is the one dependency libmlbackend cannot do
# without, and configure fails without it.
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
# LibTorch (optional): WITH_TORCH is a tri-state string, not a plain CMake
# option(), because "not requested either way, just use it if it's there" is
# a real third state here -- most environments won't have PyTorch/LibTorch
# installed, and libmlbackend must still build (ONNX-only) in that case.
#   AUTO (default): use LibTorch if find_package(Torch) succeeds, otherwise
#                   build without it.
#   ON:             require LibTorch; fail configure if it isn't found.
#   OFF:            never look for it, even if installed (e.g. to build a
#                   deliberately ONNX-only library).
# ------------------------------------------------------------------------
set(WITH_TORCH "AUTO" CACHE STRING "Whether to build with LibTorch/TorchScript support: AUTO, ON, or OFF")
set_property(CACHE WITH_TORCH PROPERTY STRINGS AUTO ON OFF)

if(NOT WITH_TORCH STREQUAL "OFF")
    find_package(Torch QUIET)
endif()

if(WITH_TORCH STREQUAL "ON" AND NOT Torch_FOUND)
    message(FATAL_ERROR "WITH_TORCH=ON but LibTorch was not found. Set -DCMAKE_PREFIX_PATH "
                         "to \"$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')\" "
                         "or use WITH_TORCH=AUTO/OFF.")
endif()

add_library(mlbackend SHARED ml_backend.cpp ml_backend.h)

target_include_directories(mlbackend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(mlbackend PUBLIC ${ONNXRUNTIME_LIBRARY})

if(Torch_FOUND)
    message(STATUS "libmlbackend: LibTorch found -- building WITH TorchScript support")
    target_compile_definitions(mlbackend PRIVATE ML_BACKEND_HAVE_TORCH)
    target_link_libraries(mlbackend PUBLIC ${TORCH_LIBRARIES})
    target_compile_options(mlbackend PRIVATE ${TORCH_CXX_FLAGS})
else()
    message(STATUS "libmlbackend: LibTorch not found (or WITH_TORCH=OFF) -- building "
                    "WITHOUT TorchScript support (ONNX only). Pointing Source_ML at a "
                    ".pt/.pth file against this build will fail at load time with an "
                    "explanatory message, not a build error.")
endif()

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
