CC = clang


TARGET = main
SRC = main.c

clean_build_and_run:
	# rm $(TARGET)
	echo "Building the out file"
	$(CC) $(SRC)\
		-I$(CONDA_PREFIX)/include\
		-L$(CONDA_PREFIX)/lib\
    -lonnxruntime \
		-Wl,-rpath,$(CONDA_PREFIX)/lib \
		-o $(TARGET)
# -L$(HOME)/onnxruntime-osx-arm64-1.26.0/lib \
	 	# -lonnxruntime \
	 	# -Wl,-rpath,$(HOME)/onnxruntime-osx-arm64-1.26.0/lib \
