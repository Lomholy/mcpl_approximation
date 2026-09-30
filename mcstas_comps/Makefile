# Thin convenience wrapper around the CMake build (CMakeLists.txt) so the
# common case is just `make` / `make install`, instead of the usual
# out-of-source CMake dance (mkdir build && cd build && cmake .. && make &&
# cmake --install .). That still happens underneath -- see README.md if you
# need more control.
#
# Respects the same environment variables as the raw CMake build:
#   CC/CXX          compiler override (e.g. for the conda-linker-vs-newer-SDK
#                   issue documented in README.md)
#   PREFIX          install prefix (default: $CONDA_PREFIX)
#
# Examples:
#   make
#   make install
#   CC=/usr/bin/clang CXX=/usr/bin/clang++ make install

BUILD_DIR := build
PREFIX ?= $(CONDA_PREFIX)

# LibTorch's CMake config dir, taken from the torch in the active Python.
TORCH_CMAKE := $(shell python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)

CMAKE_ARGS := -DCMAKE_PREFIX_PATH=$(TORCH_CMAKE) -DCMAKE_INSTALL_PREFIX=$(PREFIX)

.PHONY: all build install clean

all: build

$(BUILD_DIR)/Makefile:
	@test -n "$(PREFIX)" || { echo "PREFIX (or CONDA_PREFIX) is not set -- activate the mcpl_torch environment first, or run: make PREFIX=/path/to/env" >&2; exit 1; }
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_ARGS) ..

build: $(BUILD_DIR)/Makefile
	$(MAKE) -C $(BUILD_DIR)

install: build
	cmake --install $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
