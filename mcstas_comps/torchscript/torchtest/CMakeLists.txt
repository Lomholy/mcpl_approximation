cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(torchcall)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create shared library libtorchcall.so
add_library(torchcall SHARED
    torchcall.cpp
    torchcall.h
)

target_include_directories(torchcall PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(torchcall
    PUBLIC
    ${TORCH_LIBRARIES}
)

add_executable(main
    main.cpp
)

target_link_libraries(main
    PRIVATE
    torchcall
)


target_compile_options(torchcall PRIVATE ${TORCH_CXX_FLAGS})

target_include_directories(torchcall PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

    add_custom_command(
        TARGET torchcall
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:torchcall>
    )
endif()
