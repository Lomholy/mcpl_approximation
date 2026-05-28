CC = clang


TARGET = main
SRC = main.c

clean_build_and_run:
	# rm $(TARGET)
	echo "Building the out file"
	$(CC) $(SRC)\
	 	-I$(HOME)/onnxruntime-osx-arm64-1.26.0/include \
	 	-L$(HOME)/onnxruntime-osx-arm64-1.26.0/lib \
	 	-lonnxruntime \
	 	-Wl,-rpath,$(HOME)/onnxruntime-osx-arm64-1.26.0/lib \
		-o $(TARGET)
	./$(TARGET)
