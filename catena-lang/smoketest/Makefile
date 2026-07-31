HIPCC ?= hipcc
HIP_FLAGS ?=
BUILD_DIR ?= build
TARGET ?= $(BUILD_DIR)/array_head
SRC := src/array_head.cpp
GENERATED := ../report/gpu/hip.cpp

.PHONY: all run clean generated

all: run

generated:
	cd .. && ./check.sh examples/example.hex

$(TARGET): $(SRC) $(GENERATED)
	mkdir -p $(BUILD_DIR)
	$(HIPCC) -O2 --std=c++17 --offload-arch=native $(HIP_FLAGS) $< -o $@

run: generated $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
