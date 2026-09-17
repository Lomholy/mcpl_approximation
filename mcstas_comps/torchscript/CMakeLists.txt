cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(torch_wrap)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C-linkage bridge library: this is what Source_ML_torch.comp's DEPENDENCY
# line links against, so LibTorch's C++ API never has to be seen by the
# C-compiled component code.
add_library(torchwrap SHARED
    torch_wrap.cpp
    torch_wrap.h
)

target_include_directories(torchwrap PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(torchwrap PUBLIC ${TORCH_LIBRARIES})
target_compile_options(torchwrap PRIVATE ${TORCH_CXX_FLAGS})

# Installed into the active environment (e.g. $CONDA_PREFIX), mirroring how
# ../onnx_implementation/Source_ML.comp finds onnxruntime: Source_ML_torch.comp's
# DEPENDENCY line expects torch_wrap.h and libtorchwrap to live under
# $CONDA_PREFIX/include and $CONDA_PREFIX/lib.
install(TARGETS torchwrap LIBRARY DESTINATION lib RUNTIME DESTINATION lib)
install(FILES torch_wrap.h DESTINATION include)

# Pure-C smoke test: proves the wrapper is callable from C alone (what
# mcrun's generated instrument code will actually do).
enable_language(C)
add_executable(c_smoke_test c_smoke_test.c)
target_link_libraries(c_smoke_test PRIVATE torchwrap)
