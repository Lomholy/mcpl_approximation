CXX = g++
CXXFLAGS = -fPIC
LDFLAGS = -L.

HELLO_LIB = libhello.so
HELLOWRAP_LIB = libhellowrap.so
INSTR = Test_CppWrap.instr

.PHONY: all clean run

all: $(HELLO_LIB) $(HELLOWRAP_LIB) instrument

# Step a)
$(HELLO_LIB): hello.cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@

# Step b)
$(HELLOWRAP_LIB): hellowrap.cpp $(HELLO_LIB)
	$(CXX) $(CXXFLAGS) -shared $< $(LDFLAGS) -lhello -o $@

# Step d)
instrument: $(HELLOWRAP_LIB) $(INSTR)
	 mcrun -c $(INSTR)

# Optional target to run with the correct library path
run:
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./Test_CppWrap.out

clean:
	rm -f $(HELLO_LIB) $(HELLOWRAP_LIB)
	rm -f Test_CppWrap.c Test_CppWrap.cpp Test_CppWrap.out
	rm -rf Test_CppWrap*
